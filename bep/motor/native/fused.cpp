// Fused rollout + cost evaluation for the discovery pipeline (bep).
//
// Each rollout is simulated with the MuJoCo C API on the caller's own mjModel (shared through
// model._address) and the cost of every control step is computed in place from the sensor data:
// pair view -> relational features -> discovered relation tests -> mode bitmask -> field Phi (or
// distance to a target state) + safety + keep + leave-alone terms.  Rollouts run in parallel
// threads; nothing but the N costs goes back to Python.  The arithmetic mirrors
// bep/world/{objects,relations}.py, bep/explore/{field,primitive}.py and bep/motor/costs.py (see bep/motor/native/bridge.py, which builds
// the specification and checks it against the Python implementation).
#include <mujoco/mujoco.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

// view field order shared with native.py (VIEW_FIELDS)
enum F { EE_POS, EE_QUAT, A_POS, A_QUAT, B_POS, B_QUAT, FING1, FING2, LF_A, RF_A, A_B, A_TABLE, B_TABLE, HAND_B,
         HAND_TABLE, A_VEL, B_VEL, A_EXT, B_EXT, A_SYM, B_SYM, HAND_A, A_Q, B_Q, A_QRANGE, B_QRANGE, A_ART, B_ART,
         A_GW, B_GW, NF };
enum Kind { CC = 0, HC = 1, SC = 2 };
static const double PI = 3.14159265358979323846;
static const int ZD = 14;                      // relational state size (relations.ZDIM)

static std::vector<double> vec(const Arr& a) { return std::vector<double>(a.data(), a.data() + a.size()); }
static std::vector<int> ivec(const IArr& a) { return std::vector<int>(a.data(), a.data() + a.size()); }

struct Pair {                                  // a pair view: constants + sensor -> view copies
  std::vector<double> konst;
  std::vector<int> src, dst;
  void view(const double* sens, double* v) const {
    std::copy(konst.begin(), konst.end(), v);
    for (size_t i = 0; i < src.size(); ++i) v[dst[i]] = sens[src[i]];
  }
};

struct Relation {
  int kind, dim;
  bool body_hand;
  bool contact_only;
  bool held = false;                           // carrying grasp: both fingers on X, the line between them crosses X
  bool motion = false;                         // channel: (horizontal distance, height) of the hand (relations.motion_offset)
  double delta, tilt_max;
  std::vector<int> sig_flag;                   // flag index (per kind) and required value
  std::vector<int> sig_val;
  std::vector<double> members;                 // M x dim
  std::vector<double> center;
};

struct ModeTree {                              // compacted field samples of one mode
  std::vector<double> W, Zw, Laug;             // Zw: n x ZD (already weighted), Laug: (L - Lmin) / kappa
  double kappa, Lmin;
  int n;
};

// Phi from a learned quasimetric (bep.loop.qmap.NetField): an ensemble of MLP encoders of z, the MRN distance
// to the schema's goal embeddings, a soft minimum over them, the ensemble mean (+ unc x std).  Optional: only
// when the spec has a "net" entry; otherwise Phi comes from the compacted samples as before.
struct NetPhi {
  int E = 0, in = 0, h = 0, k2 = 0, G = 0, nb = 0;     // nb: the mode's bits appended to the input
  std::vector<double> W1, b1, W2, b2, W3, b3, mu, sd, wz, goals;
  double tau = 0.05, unc = 0.0;
  double eval(const double* z, int64_t m) const {
    std::vector<double> x(in), h1(h), h2(h), e(k2);
    for (int j = 0; j < 14; ++j) x[j] = z[j] * wz[j];
    x[14] = std::max({std::fabs(z[5]), std::fabs(z[6]), std::fabs(z[7])}) - 1.0;
    x[15] = std::max({std::fabs(z[0]), std::fabs(z[1]), std::fabs(z[2])}) - 1.0;
    for (int b = 0; b < nb; ++b) x[16 + b] = m < 0 ? 0.0 : double((m >> b) & 1);
    for (int j = 0; j < in; ++j) x[j] = (x[j] - mu[j]) / sd[j];
    const int k = k2 / 2;
    double sum = 0, sum2 = 0;
    for (int m = 0; m < E; ++m) {
      for (int o = 0; o < h; ++o) {
        double a = b1[m * h + o];
        for (int i = 0; i < in; ++i) a += x[i] * W1[(m * in + i) * h + o];
        h1[o] = a > 0 ? a : 0;
      }
      for (int o = 0; o < h; ++o) {
        double a = b2[m * h + o];
        for (int i = 0; i < h; ++i) a += h1[i] * W2[(m * h + i) * h + o];
        h2[o] = a > 0 ? a : 0;
      }
      for (int o = 0; o < k2; ++o) {
        double a = b3[m * k2 + o];
        for (int i = 0; i < h; ++i) a += h2[i] * W3[(m * h + i) * k2 + o];
        e[o] = a;
      }
      std::vector<double> d(G);
      double dmin = INFINITY;
      for (int g = 0; g < G; ++g) {
        const double* q = goals.data() + (size_t(m) * G + g) * k2;
        double s2 = 1e-8, as = 0;
        for (int j = 0; j < k; ++j) { double t = e[j] - q[j]; s2 += t * t; }
        for (int j = k; j < k2; ++j) { double t = q[j] - e[j]; if (t > as) as = t; }
        d[g] = std::sqrt(s2) + as;
        dmin = std::min(dmin, d[g]);
      }
      double acc = 0;
      for (int g = 0; g < G; ++g) acc += std::exp(-(d[g] - dmin) / tau);
      double v = dmin - tau * std::log(acc);
      sum += v; sum2 += v * v;
    }
    double mean = sum / E, var = std::max(0.0, sum2 / E - mean * mean);
    return mean + unc * std::sqrt(var);
  }
};

struct Features {
  double p[3], yaw, sym, tilt_x, tilt_y;       // object-object (A relative to B)
  bool c_xy, c_xt, c_yt;
  double hp[3], ax[3], width;                  // hand relative to A: pinch point, closing-axis code, opening
  double body_hp[3], body_ax[3];               // grasp recognition in A's full frame
  bool c_l, c_r, c_h;
  double mo[2];                                // hand relative to A folded over the direction of motion
  double qA, qB;
  bool artA, artB;
  double speedB;
};

static void quat_R(const double* q, double* R) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w); R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w); R[7] = 2 * (y * z + x * w); R[8] = 1 - 2 * (x * x + y * y);
}
static double wrap_sym(double th, double period) {
  if (period <= 0) return 0.0;
  double m = std::fmod(th + period / 2, period);
  if (m < 0) m += period;
  return m - period / 2;
}
// 1 - |cos| of the angle between vertical and the body axis closest to it (relations.tilt_of)
static double tilt_of(const double* R) {
  return 1.0 - std::min(1.0, std::max({std::fabs(R[6]), std::fabs(R[7]), std::fabs(R[8])}));
}
// closing-axis code (relations.axis_code): no sign, folded by A's symmetry about the vertical
static void axis_code(double x, double y, double zz, double sym, double* out) {
  x = std::fabs(x); y = std::fabs(y); zz = std::fabs(zz);
  if (sym <= 0) { out[0] = std::hypot(x, y); out[1] = 0.0; }
  else if (std::fabs(sym - PI / 2) < 1e-6) { out[0] = std::max(x, y); out[1] = std::min(x, y); }
  else { out[0] = x; out[1] = y; }
  out[2] = zz;
}

struct Spec {
  std::vector<int> off;                        // view field offsets (NF)
  int view_dim;
  Pair main;
  bool hc_zero;
  bool body_hand = false;
  std::vector<Relation> rels;
  std::vector<int> bit_rel, bit_role;          // mode bits: relation index, role (0 = A, 1 = B)
  std::map<int64_t, ModeTree> trees;
  bool has_pooled = false;
  ModeTree pooled;
  bool has_net = false;
  NetPhi net;
  int64_t goal_mode;
  int64_t carry_mask, carry_mode;
  bool release_goal, lift_goal;
  std::vector<int> lift_contacts;
  std::vector<double> transport_scale;
  std::vector<int> goal_slice;
  std::vector<double> goal_center, W_goal;
  double tau, cap, unexplored;
  int k;
  int cost_type;                               // 0 field, 1 distance to a target state
  int wp_resid = 0;                            // cost_type 1: goal residual instead of the full distance
  int progress_final = 0;                      // progress counted on the final step only (execution costs)
  std::vector<double> z_target, W_dist;
  int64_t m_target;
  double mode_penalty, terminal, phi_w, safety_w, rest_tilt, speed_scale, w_break, w_step, leave_w;
  std::vector<int> keep_rel;
  std::vector<Pair> keep_pair;
  std::vector<int> leave_quat, leave_vel;
  std::vector<double> leave_length;
  std::vector<int> barrier;                    // undesired contact sensors (sensordata indices)
  double barrier_w = 0, goal_reward = 0, typ_sigma = 1;
  std::vector<double> typ_W, typ_center;       // goal typicality: exploration's metric on the goal slice, centre

  double at(const double* v, F f, int i = 0) const { return v[off[f] + i]; }

  void features(const double* v, Features& f) const {
    double RA[9], RB[9], RE[9];
    quat_R(v + off[A_QUAT], RA); quat_R(v + off[B_QUAT], RB); quat_R(v + off[EE_QUAT], RE);
    double yawA = std::atan2(RA[3], RA[0]), yawB = std::atan2(RB[3], RB[0]);
    double tiltA = tilt_of(RA), tiltB = tilt_of(RB);
    const double *pA = v + off[A_POS], *pB = v + off[B_POS], *pE = v + off[EE_POS];
    const double *eA = v + off[A_EXT], *eB = v + off[B_EXT];
    double dx = pA[0] - pB[0], dy = pA[1] - pB[1], dz = pA[2] - pB[2];
    double c = std::cos(yawB), s = std::sin(yawB);
    double rX = std::max(eA[0], eA[1]);
    f.p[0] = (c * dx + s * dy) / (eB[0] + rX);
    f.p[1] = (-s * dx + c * dy) / (eB[1] + rX);
    f.p[2] = dz / (eA[2] + eB[2]);
    f.sym = at(v, A_SYM);
    f.yaw = wrap_sym(yawA - yawB, f.sym);
    f.c_xy = at(v, A_B) > 0; f.c_xt = at(v, A_TABLE) > 0; f.c_yt = at(v, B_TABLE) > 0;
    f.tilt_x = tiltA; f.tilt_y = tiltB;
    double ex = pE[0] - pA[0], ey = pE[1] - pA[1], ez = pE[2] - pA[2];
    double ca = std::cos(yawA), sa = std::sin(yawA);
    f.hp[0] = (ca * ex + sa * ey) / eA[0];
    f.hp[1] = (-sa * ex + ca * ey) / eA[1];
    f.hp[2] = ez / eA[2];
    // closing axis = column 1 of the hand rotation (row-major RE[1], RE[4], RE[7]), in A's yaw frame
    axis_code(ca * RE[1] + sa * RE[4], -sa * RE[1] + ca * RE[4], RE[7], f.sym, f.ax);
    if (body_hand) {
      for (int j = 0; j < 3; ++j)
        f.body_hp[j] = (RA[j] * ex + RA[j + 3] * ey + RA[j + 6] * ez) / eA[j];
      const double px = std::fabs(f.body_hp[0]), py = std::fabs(f.body_hp[1]);
      if (f.sym <= 0) {
        f.body_hp[0] = std::hypot(px, py); f.body_hp[1] = 0.0;
      } else if (std::fabs(f.sym - PI / 2) < 1e-6) {
        f.body_hp[0] = std::max(px, py); f.body_hp[1] = std::min(px, py);
      } else if (std::fabs(f.sym - PI) < 1e-6) {
        f.body_hp[0] = px; f.body_hp[1] = py;
      }
      double a[3];
      for (int j = 0; j < 3; ++j) a[j] = RA[j] * RE[1] + RA[j + 3] * RE[4] + RA[j + 6] * RE[7];
      axis_code(a[0], a[1], a[2], f.sym, f.body_ax);
    }
    f.width = (at(v, FING1) + at(v, FING2)) / at(v, A_GW);
    f.c_l = at(v, LF_A) > 0; f.c_r = at(v, RF_A) > 0; f.c_h = at(v, HAND_A) > 0;
    {
      const double L = 2.0 * std::cbrt(eA[0] * eA[1] * eA[2]);    // geometry.characteristic_length
      f.mo[0] = std::hypot(ex, ey) / L; f.mo[1] = ez / L;
    }
    f.artA = at(v, A_ART) > 0; f.artB = at(v, B_ART) > 0;
    f.qA = f.artA ? (at(v, A_Q) - at(v, A_QRANGE, 0)) / std::max(at(v, A_QRANGE, 1) - at(v, A_QRANGE, 0), 1e-9) : 0.0;
    f.qB = f.artB ? (at(v, B_Q) - at(v, B_QRANGE, 0)) / std::max(at(v, B_QRANGE, 1) - at(v, B_QRANGE, 0), 1e-9) : 0.0;
    const double* vB = v + off[B_VEL];
    f.speedB = std::sqrt(vB[0] * vB[0] + vB[1] * vB[1] + vB[2] * vB[2]);
  }

  static void code(double th, double period, double* out) {
    double k = period > 0 ? 2 * PI / period : 0.0;
    out[0] = std::cos(k * th) / 4; out[1] = std::sin(k * th) / 4;
  }

  void zvec(const Features& f, double* z) const {
    z[0] = f.p[0]; z[1] = f.p[1]; z[2] = f.p[2];
    code(f.yaw, f.sym, z + 3);
    z[5] = f.hp[0]; z[6] = f.hp[1]; z[7] = f.hp[2];
    z[8] = f.ax[0]; z[9] = f.ax[1]; z[10] = f.ax[2];
    z[11] = f.width; z[12] = f.qA; z[13] = f.qB;
    if (hc_zero) { z[0] = z[1] = 0.0; z[3] = z[4] = 0.0; }
  }

  bool holds(const Relation& r, const Features& f, int role) const {
    double x[7];
    if (r.kind == CC) {
      bool fl[3] = {f.c_xy, f.c_xt, f.c_yt};
      for (size_t i = 0; i < r.sig_flag.size(); ++i) if (fl[r.sig_flag[i]] != (r.sig_val[i] != 0)) return false;
      if (f.tilt_x > r.tilt_max || f.tilt_y > r.tilt_max) return false;
      x[0] = f.p[0]; x[1] = f.p[1]; x[2] = f.p[2];
    } else if (r.kind == HC) {
      bool fl[5] = {f.c_l, f.c_r, f.c_h, f.c_l && f.c_r, f.c_l || f.c_r || f.c_h};
      for (size_t i = 0; i < r.sig_flag.size(); ++i) if (fl[r.sig_flag[i]] != (r.sig_val[i] != 0)) return false;
      if (r.contact_only) return true;
      if (r.motion) { x[0] = f.mo[0]; x[1] = f.mo[1]; }
      else if (r.held)                                // relations.pinch_outside == 0
        return std::max({std::fabs(f.body_hp[0]), std::fabs(f.body_hp[1]), std::fabs(f.body_hp[2])}) <= 1.0;
      else {
        const double* p = r.body_hand ? f.body_hp : f.hp;
        const double* a = r.body_hand ? f.body_ax : f.ax;
        std::copy(p, p + 3, x); std::copy(a, a + 3, x + 3); x[6] = f.width;
      }
    } else {
      bool art = role == 0 ? f.artA : f.artB;
      for (size_t i = 0; i < r.sig_flag.size(); ++i) if (art != (r.sig_val[i] != 0)) return false;
      x[0] = role == 0 ? f.qA : f.qB;
    }
    double d2 = r.delta * r.delta;
    size_t M = r.members.size() / r.dim;
    for (size_t m = 0; m < M; ++m) {
      const double* y = r.members.data() + m * r.dim;
      double s = 0;
      for (int j = 0; j < r.dim; ++j) { double d = x[j] - y[j]; s += d * d; if (s > d2) break; }
      if (s <= d2) return true;
    }
    return false;
  }

  int64_t mode(const Features& f) const {
    int64_t m = 0;
    for (size_t b = 0; b < bit_rel.size(); ++b)
      if (holds(rels[bit_rel[b]], f, bit_role[b])) m |= (int64_t(1) << b);
    return m;
  }

  double softmin(const ModeTree& t, const double* z) const {
    int kk = std::min(k, t.n);
    std::vector<double> best(kk, INFINITY);
    for (int i = 0; i < t.n; ++i) {
      const double* y = t.Zw.data() + i * ZD;
      double d = t.Laug[i];
      for (int j = 0; j < ZD; ++j) d += std::fabs(z[j] * t.W[j] - y[j]);
      if (d < best[kk - 1]) {                    // keep the kk smallest (insertion)
        int pos = kk - 1;
        while (pos > 0 && best[pos - 1] > d) { best[pos] = best[pos - 1]; --pos; }
        best[pos] = d;
      }
    }
    double vmin = t.Lmin + t.kappa * best[0], acc = 0;
    for (int i = 0; i < kk; ++i) acc += std::exp(-((t.Lmin + t.kappa * best[i]) - vmin) / tau);
    return vmin - tau * std::log(acc);
  }

  double phi(const double* z, int64_t m) const {
    double out = cap;
    if (m == goal_mode) {
      if (lift_goal) return 0.0;
      double r = 0;
      for (size_t i = 0; i < goal_slice.size(); ++i) r += std::fabs(z[goal_slice[i]] - goal_center[i]) * W_goal[i];
      out = r;
    } else if (has_net) {
      out = net.eval(z, m);
    } else {
      auto it = trees.find(m);
      if (it != trees.end()) out = softmin(it->second, z);
      else if (has_pooled) out = softmin(pooled, z) + unexplored;
    }
    return std::min(out, cap);
  }

  double step_cost(const double* sens, std::vector<double>& v, double& last_phi, double& goal_acc,
                   bool final_step) const {
    Features f;
    main.view(sens, v.data());
    features(v.data(), f);
    const bool progress = !progress_final || final_step;
    int64_t m = progress || goal_reward > 0 || carry_mode ? mode(f) : 0;
    const bool pinching = f.c_l && f.c_r;
    if (lift_goal && f.c_xt) m &= ~goal_mode;
    if ((m & goal_mode) && (!release_goal || (!(m & carry_mask) && !pinching))) m = goal_mode;
    if (carry_mode) {
      if (m & goal_mode) m = goal_mode;
      else if (pinching && !(m & carry_mask)) m |= carry_mode;
    }
    bool reward_goal = goal_reward > 0 && m == goal_mode && (carry_mode || !release_goal || !pinching);
    if (reward_goal)
      for (int i : lift_contacts) if (sens[i] > 0) { reward_goal = false; break; }
    double z[ZD];
    if (progress || reward_goal) zvec(f, z);
    if (reward_goal) {                              // reward typical goal instances throughout the rollout
      double r = 0;
      for (size_t i = 0; i < goal_slice.size(); ++i) r += std::fabs(z[goal_slice[i]] - typ_center[i]) * typ_W[i];
      goal_acc += lift_goal ? 1.0 : std::exp(-r / typ_sigma);
    }
    double safety = safety_w * (f.tilt_y / rest_tilt + f.speedB / speed_scale);
    double c = safety;
    if (progress && cost_type == 0) {
      double ph = phi(z, m);
      if (carry_mode) {
        double d2 = 0.0;
        for (size_t i = 0; i < goal_slice.size(); ++i) {
          double error = (z[goal_slice[i]] - goal_center[i]) * transport_scale[i];
          d2 += error * error;
        }
        ph += std::sqrt(d2);
      }
      last_phi = ph;
      if (!progress_final) c = ph * phi_w + safety;
    } else if (progress) {
      double d = 0;
      if (wp_resid) {          // 1: relation residual (exploration's goal targets); 2: + the hand's opening
        for (size_t i = 0; i < goal_slice.size(); ++i) d += std::fabs(z[goal_slice[i]] - goal_center[i]);
        if (wp_resid == 2 && !carry_mode) d += W_dist[11] * std::fabs(z[11] - z_target[11]);
      } else {
        double s2 = 0;
        for (int j = 0; j < ZD; ++j) { double e = (z[j] - z_target[j]) * W_dist[j]; s2 += e * e; }
        d = std::sqrt(s2);
      }
      double prog = d + mode_penalty * (m != m_target ? 1.0 : 0.0);
      c = progress_final ? safety : prog + safety;
      last_phi = progress_final ? prog : c;
    }
    for (size_t i = 0; i < keep_rel.size(); ++i) {    // keep every relation that should persist
      const Relation& r = rels[keep_rel[i]];
      Features g;
      keep_pair[i].view(sens, v.data());
      features(v.data(), g);
      bool h = holds(r, g, 0);
      double res = 0;
      if (r.kind == SC) res = std::fabs(g.qA - r.center[0]);
      else for (int j = 0; j < 3; ++j) res += std::fabs(g.p[j] - r.center[j]);
      c += w_break * (h ? 0.0 : 1.0) + w_step * res;
    }
    for (size_t i = 0; i < leave_quat.size(); ++i) {  // leave every other body alone
      double R[9];
      quat_R(sens + leave_quat[i], R);
      double tilt = tilt_of(R);
      const double* vv = sens + leave_vel[i];
      double sp = std::sqrt(vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]);
      c += leave_w * (tilt / rest_tilt + sp / leave_length[i]);
    }
    for (int b : barrier) if (sens[b] > 0) c += barrier_w;   // contact barrier
    if (carry_mode && !pinching && m != goal_mode) c += w_break;
    return c;
  }
};

static Relation make_rel(py::dict d) {
  Relation r;
  r.kind = d["kind"].cast<int>();
  r.dim = d["dim"].cast<int>();
  r.body_hand = d["body_hand"].cast<bool>();
  r.contact_only = d["contact_only"].cast<bool>();
  r.held = d["held"].cast<bool>();
  r.motion = d.contains("motion") && d["motion"].cast<bool>();
  r.delta = d["delta"].cast<double>();
  r.tilt_max = d["tilt_max"].cast<double>();
  r.sig_flag = ivec(d["sig_flag"].cast<IArr>());
  r.sig_val = ivec(d["sig_val"].cast<IArr>());
  r.members = vec(d["members"].cast<Arr>());
  r.center = vec(d["center"].cast<Arr>());
  return r;
}
static ModeTree make_tree(py::dict d) {
  ModeTree t;
  t.W = vec(d["W"].cast<Arr>()); t.Zw = vec(d["Zw"].cast<Arr>()); t.Laug = vec(d["Laug"].cast<Arr>());
  t.kappa = d["kappa"].cast<double>(); t.Lmin = d["Lmin"].cast<double>();
  t.n = int(t.Laug.size());
  return t;
}
static Pair make_pair(py::dict d) {
  Pair p;
  p.konst = vec(d["const"].cast<Arr>()); p.src = ivec(d["src"].cast<IArr>()); p.dst = ivec(d["dst"].cast<IArr>());
  return p;
}

// persistent worker threads: run(f) calls f(tid) for tid = 0..n-1 (0 on the caller) and waits
class Pool {
 public:
  explicit Pool(int n) : n_(n) {
    for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { loop(i); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> g(mu_); stop_ = true; ++gen_; }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  void run(const std::function<void(int)>& f) {
    { std::lock_guard<std::mutex> g(mu_); job_ = &f; pending_ = n_ - 1; ++gen_; }
    cv_.notify_all();
    f(0);
    std::unique_lock<std::mutex> l(mu_);
    done_.wait(l, [this] { return pending_ == 0; });
  }
 private:
  void loop(int i) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* f;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        f = job_;
      }
      (*f)(i);
      { std::lock_guard<std::mutex> g(mu_); if (--pending_ == 0) done_.notify_one(); }
    }
  }
  int n_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* job_ = nullptr;
  int pending_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

class Fused {
 public:
  explicit Fused(int nthread) : nthread_(std::max(1, nthread)), pool_(std::max(1, nthread)) {}
  ~Fused() {
    for (auto& kv : datas_) for (auto* d : kv.second) mj_deleteData(d);
    for (auto& kv : kin_) mj_deleteData(kv.second);
  }

  // Spike-MPPI nominal plan: integrate the Cartesian velocity plan U through the damped pseudo-inverse of
  // the pinch-site Jacobian with posture terms (mirrors bep.motor.mppi.SpikeMPPI._nominal_maps).  U is
  // (H, 4: vx vy vz wz), roll and pitch servoed to pointing down by ori_gain, or (H, 6: vx vy vz wz wx wy),
  // 6-DoF.  Returns P (H, 7, 4 or 6) and the joint targets Qn (H, 7).
  py::tuple nominal_maps(std::uintptr_t model_addr, Arr q_base, Arr U, IArr qadr, IArr dadr, int sid, double ori_gain,
                         double posture_gain, double dt, Arr low, Arr high, Arr home, double manip_gain) {
    const mjModel* m = reinterpret_cast<const mjModel*>(model_addr);
    auto it = kin_.find(m);
    if (it == kin_.end()) it = kin_.emplace(m, mj_makeData(m)).first;
    mjData* d = it->second;
    int H = int(U.shape(0)), nc = int(U.shape(1)), nv = m->nv;
    const bool free_tilt = nc == 6;
    const double* u = U.data();
    const int* qa = qadr.data(); const int* da = dadr.data();
    py::array_t<double> P({H, 7, nc}), Qn({H, 7});
    double* pp = P.mutable_data(); double* qn = Qn.mutable_data();
    double q[7];
    std::copy(q_base.data(), q_base.data() + 7, q);
    std::vector<double> jp(3 * nv), jr(3 * nv);
    for (int t = 0; t < H; ++t) {
      for (int i = 0; i < 7; ++i) d->qpos[qa[i]] = q[i];
      mj_kinematics(m, d);
      mj_comPos(m, d);
      const double* R = d->site_xmat + 9 * sid;                // row-major 3x3
      double yaw = std::atan2(R[3], R[0]), c = std::cos(yaw), s = std::sin(yaw);
      double Rd[9] = {c, s, 0, s, -c, 0, 0, 0, -1};            // ee_down_rot(yaw), row-major
      double er[3] = {0, 0, 0};
      for (int k = 0; k < 3; ++k) {                            // 0.5 * sum_k col_k(R) x col_k(Rd)
        double a0 = R[k], a1 = R[3 + k], a2 = R[6 + k], b0 = Rd[k], b1 = Rd[3 + k], b2 = Rd[6 + k];
        er[0] += 0.5 * (a1 * b2 - a2 * b1); er[1] += 0.5 * (a2 * b0 - a0 * b2); er[2] += 0.5 * (a0 * b1 - a1 * b0);
      }
      mj_jacSite(m, d, jp.data(), jr.data(), sid);
      double J[42];
      for (int r = 0; r < 3; ++r)
        for (int j = 0; j < 7; ++j) { J[r * 7 + j] = jp[r * nv + da[j]]; J[(r + 3) * 7 + j] = jr[r * nv + da[j]]; }
      double A[36];
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) {
          double acc = a == b ? 1e-4 : 0.0;
          for (int j = 0; j < 7; ++j) acc += J[a * 7 + j] * J[b * 7 + j];
          A[a * 6 + b] = acc;
        }
      mju_cholFactor(A, 6, 0);
      double Ai[36];                                           // A^-1 (symmetric)
      for (int col = 0; col < 6; ++col) {
        double e[6] = {0, 0, 0, 0, 0, 0}, x[6];
        e[col] = 1.0;
        mju_cholSolve(x, A, e, 6);
        for (int r = 0; r < 6; ++r) Ai[r * 6 + col] = x[r];
      }
      double Jp[42];                                           // J^T A^-1 (7x6)
      for (int j = 0; j < 7; ++j)
        for (int b = 0; b < 6; ++b) {
          double acc = 0;
          for (int a = 0; a < 6; ++a) acc += J[a * 7 + j] * Ai[a * 6 + b];
          Jp[j * 6 + b] = acc;
        }
      const int cols4[4] = {0, 1, 2, 5}, cols6[6] = {0, 1, 2, 5, 3, 4};
      const int* cols = free_tilt ? cols6 : cols4;
      for (int j = 0; j < 7; ++j)
        for (int k = 0; k < nc; ++k) pp[(t * 7 + j) * nc + k] = Jp[j * 6 + cols[k]];
      const double* ut = u + t * nc;
      double tw[6] = {ut[0], ut[1], ut[2], ori_gain * er[0], ori_gain * er[1], ut[3]};
      if (free_tilt) { tw[3] = ut[4]; tw[4] = ut[5]; }            // roll and pitch rates are commanded
      // null space: pull to the home posture, and optionally ascend log det(J J^T) of the pinch Jacobian
      // (finite differences; mirrors SpikeMPPI)
      double post[7], Jpost[6];
      for (int k = 0; k < 7; ++k) post[k] = posture_gain * (home.data()[k] - q[k]);
      if (manip_gain > 0) {
        auto logdet = [&](const double* qq) {
          for (int k = 0; k < 7; ++k) d->qpos[qa[k]] = qq[k];
          mj_kinematics(m, d);
          mj_comPos(m, d);
          mj_jacSite(m, d, jp.data(), jr.data(), sid);
          double Jl[42], M[36];
          for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 7; ++k) { Jl[r * 7 + k] = jp[r * nv + da[k]]; Jl[(r + 3) * 7 + k] = jr[r * nv + da[k]]; }
          for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
              double acc = a == b ? 1e-6 : 0.0;
              for (int k = 0; k < 7; ++k) acc += Jl[a * 7 + k] * Jl[b * 7 + k];
              M[a * 6 + b] = acc;
            }
          mju_cholFactor(M, 6, 0);
          double ld = 0;
          for (int a = 0; a < 6; ++a) ld += 2 * std::log(M[a * 6 + a]);
          return ld;
        };
        const double h = 1e-3;
        double f0 = logdet(q), qq[7];
        for (int k = 0; k < 7; ++k) {
          std::copy(q, q + 7, qq);
          qq[k] += h;
          post[k] += manip_gain * (logdet(qq) - f0) / h;
        }
        logdet(q);                                             // leave d at q (and J's data as before)
      }
      for (int a = 0; a < 6; ++a) { Jpost[a] = 0; for (int j = 0; j < 7; ++j) Jpost[a] += J[a * 7 + j] * post[j]; }
      for (int j = 0; j < 7; ++j) {
        double dq = post[j];
        for (int b = 0; b < 6; ++b) dq += Jp[j * 6 + b] * (tw[b] - Jpost[b]);
        q[j] = std::min(std::max(q[j] + dq * dt, low.data()[j]), high.data()[j]);
        qn[t * 7 + j] = q[j];
      }
    }
    return py::make_tuple(P, Qn);
  }

  void set_spec(py::dict d) {
    Spec s;
    s.off = ivec(d["off"].cast<IArr>());
    s.view_dim = d["view_dim"].cast<int>();
    s.main = make_pair(d["main"].cast<py::dict>());
    s.hc_zero = d["hc_zero"].cast<bool>();
    for (auto r : d["rels"].cast<py::list>()) {
      s.rels.push_back(make_rel(r.cast<py::dict>()));
      s.body_hand = s.body_hand || s.rels.back().body_hand;
    }
    s.bit_rel = ivec(d["bit_rel"].cast<IArr>()); s.bit_role = ivec(d["bit_role"].cast<IArr>());
    for (auto item : d["trees"].cast<py::dict>()) s.trees[item.first.cast<int64_t>()] = make_tree(item.second.cast<py::dict>());
    if (!d["pooled"].is_none()) { s.has_pooled = true; s.pooled = make_tree(d["pooled"].cast<py::dict>()); }
    if (d.contains("net") && !d["net"].is_none()) {
      py::dict n = d["net"].cast<py::dict>();
      s.has_net = true;
      s.net.E = n["E"].cast<int>(); s.net.in = n["in"].cast<int>(); s.net.h = n["h"].cast<int>();
      s.net.k2 = n["k2"].cast<int>(); s.net.G = n["G"].cast<int>();
      s.net.nb = n.contains("nb") ? n["nb"].cast<int>() : 0;
      s.net.W1 = vec(n["W1"].cast<Arr>()); s.net.b1 = vec(n["b1"].cast<Arr>());
      s.net.W2 = vec(n["W2"].cast<Arr>()); s.net.b2 = vec(n["b2"].cast<Arr>());
      s.net.W3 = vec(n["W3"].cast<Arr>()); s.net.b3 = vec(n["b3"].cast<Arr>());
      s.net.mu = vec(n["mu"].cast<Arr>()); s.net.sd = vec(n["sd"].cast<Arr>()); s.net.wz = vec(n["wz"].cast<Arr>());
      s.net.goals = vec(n["goals"].cast<Arr>());
      s.net.tau = n["tau"].cast<double>(); s.net.unc = n["unc"].cast<double>();
    }
    s.goal_mode = d["goal_mode"].cast<int64_t>();
    s.carry_mask = d["carry_mask"].cast<int64_t>();
    s.carry_mode = d["carry_mode"].cast<int64_t>();
    s.release_goal = d["release_goal"].cast<bool>();
    s.lift_goal = d["lift_goal"].cast<bool>();
    s.lift_contacts = ivec(d["lift_contacts"].cast<IArr>());
    s.transport_scale = vec(d["transport_scale"].cast<Arr>());
    s.goal_slice = ivec(d["goal_slice"].cast<IArr>());
    s.goal_center = vec(d["goal_center"].cast<Arr>()); s.W_goal = vec(d["W_goal"].cast<Arr>());
    s.unexplored = d["unexplored"].cast<double>();
    s.tau = d["tau"].cast<double>(); s.cap = d["cap"].cast<double>(); s.k = d["k"].cast<int>();
    s.cost_type = d["cost_type"].cast<int>();
    s.wp_resid = d["wp_resid"].cast<int>();
    s.progress_final = d["progress_final"].cast<int>();
    s.z_target = vec(d["z_target"].cast<Arr>()); s.W_dist = vec(d["W_dist"].cast<Arr>());
    s.m_target = d["m_target"].cast<int64_t>();
    s.mode_penalty = d["mode_penalty"].cast<double>(); s.terminal = d["terminal"].cast<double>();
    s.phi_w = d["phi_w"].cast<double>(); s.safety_w = d["safety_w"].cast<double>();
    s.rest_tilt = d["rest_tilt"].cast<double>(); s.speed_scale = d["speed_scale"].cast<double>();
    s.w_break = d["w_break"].cast<double>(); s.w_step = d["w_step"].cast<double>();
    s.leave_w = d["leave_w"].cast<double>();
    s.keep_rel = ivec(d["keep_rel"].cast<IArr>());
    for (auto p : d["keep_pair"].cast<py::list>()) s.keep_pair.push_back(make_pair(p.cast<py::dict>()));
    s.leave_quat = ivec(d["leave_quat"].cast<IArr>()); s.leave_vel = ivec(d["leave_vel"].cast<IArr>());
    s.leave_length = vec(d["leave_length"].cast<Arr>());
    s.barrier = ivec(d["barrier"].cast<IArr>()); s.barrier_w = d["barrier_w"].cast<double>();
    s.goal_reward = d["goal_reward"].cast<double>(); s.typ_sigma = d["typ_sigma"].cast<double>();
    s.typ_W = vec(d["typ_W"].cast<Arr>()); s.typ_center = vec(d["typ_center"].cast<Arr>());
    spec_ = std::move(s);
  }

  // controls: (N, H, nc) one control per control step; each held for ``substeps`` physics steps.  nc = nu, or
  // with ``mocap`` (the floating hand) nu + 7 nmocap: ctrl, mocap_pos, mocap_quat (mujoco.rollout's
  // CTRL | MOCAP_POS | MOCAP_QUAT control layout).
  py::array_t<double> evaluate(std::uintptr_t model_addr, Arr x0, Arr controls, int substeps, bool mocap) {
    const mjModel* m = reinterpret_cast<const mjModel*>(model_addr);
    auto c = controls.unchecked<3>();
    int N = int(c.shape(0)), H = int(c.shape(1)), nu = int(c.shape(2));
    const int mu = m->nu, nmp = 3 * m->nmocap, nmq = 4 * m->nmocap;
    if (nu != (mocap ? mu + nmp + nmq : mu)) throw std::invalid_argument("controls do not match the model");
    std::vector<double> state(x0.data(), x0.data() + x0.size());
    std::vector<mjData*>& ds = datas(m);
    py::array_t<double> out(N);
    double* o = out.mutable_data();
    const double* cd = controls.data();
    {
      py::gil_scoped_release release;
      auto work = [&](int tid) {
        mjData* d = ds[tid];
        std::vector<double> v(spec_.view_dim);
        for (int n = tid; n < N; n += nthread_) {
          mj_setState(m, d, state.data(), mjSTATE_FULLPHYSICS);
          double total = 0, last = 0, goal_acc = 0;
          for (int t = 0; t < H; ++t) {
            const double* row = cd + (size_t(n) * H + t) * nu;
            std::copy(row, row + mu, d->ctrl);
            if (mocap) {
              std::copy(row + mu, row + mu + nmp, d->mocap_pos);
              std::copy(row + mu + nmp, row + mu + nmp + nmq, d->mocap_quat);
            }
            for (int s = 0; s < substeps; ++s) mj_step(m, d);
            total += spec_.step_cost(d->sensordata, v, last, goal_acc, t + 1 == H);
          }
          o[n] = finish(total, last, goal_acc, H);
        }
      };
      pool_.run(work);
    }
    return out;
  }

  // a rollout's total from its per-step costs (shared by evaluate and costs_of): the goal reward, and the
  // terminal: the last step's field value (field cost) or its full per-step cost (target cost), weighted by
  // the terminal weight plus (progress on the final step only) the per-step progress weight x H (phi_w for
  // the field, 1 for a waypoint)
  double finish(double total, double last, double goal_acc, int H) const {
    total -= spec_.goal_reward * goal_acc / H;
    double wf = spec_.terminal + (spec_.progress_final ? (spec_.cost_type == 0 ? spec_.phi_w : 1.0) * H : 0.0);
    double o = total + wf * last;
    return std::isfinite(o) ? o : INFINITY;
  }

  // the rollout costs of given sensor rows (N, H, nsensordata): evaluate's cost without the physics (checks
  // against the Python implementation on identical input)
  py::array_t<double> costs_of(Arr sens) {
    auto s = sens.unchecked<3>();
    int N = int(s.shape(0)), H = int(s.shape(1)), ns = int(s.shape(2));
    py::array_t<double> out(N);
    std::vector<double> v(spec_.view_dim);
    for (int n = 0; n < N; ++n) {
      double total = 0, last = 0, goal_acc = 0;
      for (int t = 0; t < H; ++t)
        total += spec_.step_cost(sens.data() + (size_t(n) * H + t) * ns, v, last, goal_acc, t + 1 == H);
      out.mutable_at(n) = finish(total, last, goal_acc, H);
    }
    return out;
  }

  // Phi of given relational states (N, 14) in given modes (checks against the Python implementation)
  py::array_t<double> phi_of(Arr z, py::array_t<int64_t> modes) {
    auto zz = z.unchecked<2>();
    py::array_t<double> out(zz.shape(0));
    for (py::ssize_t i = 0; i < zz.shape(0); ++i) out.mutable_at(i) = spec_.phi(z.data() + i * 14, modes.at(i));
    return out;
  }

  // the per-step cost of given sensor rows (checks against the Python implementation)
  py::array_t<double> step_costs(Arr sens) {
    auto s = sens.unchecked<2>();
    py::array_t<double> out(s.shape(0));
    std::vector<double> v(spec_.view_dim);
    double last, goal_acc = 0;
    for (py::ssize_t i = 0; i < s.shape(0); ++i)
      out.mutable_at(i) = spec_.step_cost(sens.data() + i * s.shape(1), v, last, goal_acc, true);
    return out;
  }

 private:
  std::vector<mjData*>& datas(const mjModel* m) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = datas_.find(m);
    if (it == datas_.end()) {
      std::vector<mjData*> ds;
      for (int i = 0; i < nthread_; ++i) ds.push_back(mj_makeData(m));
      it = datas_.emplace(m, ds).first;
    }
    return it->second;
  }
  int nthread_;
  Pool pool_;
  Spec spec_;
  std::map<const mjModel*, std::vector<mjData*>> datas_;
  std::map<const mjModel*, mjData*> kin_;
  std::mutex mu_;
};

PYBIND11_MODULE(_fused, mod) {
  py::class_<Fused>(mod, "Fused")
      .def(py::init<int>())
      .def("set_spec", &Fused::set_spec)
      .def("evaluate", &Fused::evaluate, py::arg("model"), py::arg("x0"), py::arg("controls"), py::arg("substeps"),
           py::arg("mocap") = false)
      .def("step_costs", &Fused::step_costs)
      .def("costs_of", &Fused::costs_of)
      .def("phi_of", &Fused::phi_of)
      .def("nominal_maps", &Fused::nominal_maps);
}
