#include "camera_tracking.hpp"
#include <iostream>
#include <stdexcept>

void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        CameraTracking live;
        ArmorEKF reference{};
        Eigen::Matrix<double, 11, 1> state;
        state << .1,0,.2,0,3,0,0,0,.26,0,0;
        const auto observation = armor_h(state,0);
        live.update({},0);
        check(!live.visible() && !live.command_.control_valid, "empty first frame");
        bool duplicate_zero = false;
        try { live.update({}, 0); }
        catch (const std::invalid_argument &) { duplicate_zero = true; }
        check(duplicate_zero, "live time zero is a valid previous timestamp");
        CameraTracking zero_seed;
        zero_seed.update({{observation}}, 0);
        zero_seed.update({}, 201);
        check(zero_seed.status_ == "lost", "observation at time zero participates in loss timing");
        live.update({{observation}},30);
        reference.initialize(observation);
        check(live.visible() && live.status_=="initializing", "initialization");
        for (int frame=2; frame<10; ++frame) {
            std::vector<Observation> obs{{armor_h(state,0)}, {armor_h(state,1)}};
            reference.predict(.03);
            reference.update_multi(obs);
            live.update(obs,frame*30.);
            check((reference.X-live.ekf_.X).norm()<1e-12, "live state differs from EKF");
            check((reference.P-live.ekf_.P).norm()<1e-12, "live covariance differs from EKF");
        }
        live.update({},300);
        check(live.visible() && live.status_=="prediction_only", "short dropout");
        check(!live.command_.control_valid && live.command_.yaw_rate_dps==0, "dropout must hold");
        live.update({},510);
        check(!live.visible() && live.status_=="lost", "stale prediction not hidden");
        bool rejected=false;
        try { live.update({},510); } catch(const std::invalid_argument &) { rejected=true; }
        check(rejected,"duplicate timestamp");
        CameraTracking manual;
        manual.key('1');
        for (int i=0;i<5;++i) { manual.key('d'); manual.key('w'); }
        for (int i=0;i<200;++i) manual.update({},i*30.);
        check(manual.mode_==TRACKING_MANUAL, "manual mode key");
        check((manual.simulated_angles_-Eigen::Vector2d(10,10)).norm()<1e-9,
              "manual reaches requested position without observations");
        manual.key('c');
        for (int i=200;i<500;++i) manual.update({},i*30.);
        check(manual.mode_==TRACKING_CENTER && manual.simulated_angles_.norm()<1e-9,
              "center returns both axes to startup zero");
        manual.key('2');
        manual.update({},15000);
        check(manual.mode_==TRACKING_AUTOMATIC && !manual.command_.control_valid,
              "auto without observations holds");
        CameraTracking automatic;
        for (int i=0;i<150;++i) automatic.update({{observation}},i*30.);
        check(automatic.simulated_angles_.norm()>0.1,"auto moves simulated platform");
        const auto forecast=automatic.ekf_.forecast(camera_tracking_horizon_s);
        double nearest=1e9;
        for (int i=0;i<4;++i)
            nearest=std::min(nearest,(forecast.plates.row(i).head(3).transpose()-automatic.predicted_target_).norm());
        check(nearest<1e-12,"auto target is a forecast plate");
        const auto held=automatic.simulated_angles_;
        automatic.update({},4500);
        check((automatic.simulated_angles_-held).norm()==0,"loss holds simulated position");
        std::cout << "Live camera tracking checks passed\n";
    } catch(const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
