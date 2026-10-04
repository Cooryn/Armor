#include "camera_tracking.hpp"
#include <iostream>
#include <stdexcept>

void check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        CameraTracking live;
        ArmorEKF reference{};
        ArmorState state;
        state << .1,0,.2,0,3,0,0,0,.26,0,0;
        const auto observation = armor_h(state,0);
        camera_tracking_update(live, {},0);
        check(!camera_tracking_visible(live) && !live.command_.control_valid, "empty first frame");
        bool duplicate_zero = false;
        try { camera_tracking_update(live, {}, 0); }
        catch (const std::invalid_argument &) { duplicate_zero = true; }
        check(duplicate_zero, "live time zero is a valid previous timestamp");
        CameraTracking zero_seed;
        camera_tracking_update(zero_seed, {{observation}}, 0);
        camera_tracking_update(zero_seed, {}, 201);
        check(zero_seed.status_ == "lost", "observation at time zero participates in loss timing");
        camera_tracking_update(live, {{observation}},30);
        armor_initialize(reference, observation);
        check(camera_tracking_visible(live) && live.status_=="initializing", "initialization");
        for (int frame=2; frame<10; ++frame) {
            std::vector<Observation> obs{{armor_h(state,0)}, {armor_h(state,1)}};
            armor_predict(reference, .03);
            armor_update_multi(reference, obs);
            camera_tracking_update(live, obs,frame*30.);
            check((reference.X-live.ekf_.X).norm()<1e-12, "live state differs from EKF");
            check((reference.P-live.ekf_.P).norm()<1e-12, "live covariance differs from EKF");
        }
        camera_tracking_update(live, {},300);
        check(camera_tracking_visible(live) && live.status_=="prediction_only", "short dropout");
        check(!live.command_.control_valid && live.command_.yaw_rate_dps==0, "dropout must hold");
        camera_tracking_update(live, {},510);
        check(!camera_tracking_visible(live) && live.status_=="lost", "stale prediction not hidden");
        bool rejected=false;
        try { camera_tracking_update(live, {},510); } catch(const std::invalid_argument &) { rejected=true; }
        check(rejected,"duplicate timestamp");
        CameraTracking manual;
        camera_tracking_key(manual, '1');
        for (int i=0;i<5;++i) { camera_tracking_key(manual, 'd'); camera_tracking_key(manual, 'w'); }
        for (int i=0;i<200;++i) camera_tracking_update(manual, {},i*30.);
        check(manual.mode_==TRACKING_MANUAL, "manual mode key");
        check((manual.simulated_angles_-Eigen::Vector2d(10,10)).norm()<1e-9,
              "manual reaches requested position without observations");
        camera_tracking_key(manual, 'c');
        for (int i=200;i<500;++i) camera_tracking_update(manual, {},i*30.);
        check(manual.mode_==TRACKING_CENTER && manual.simulated_angles_.norm()<1e-9,
              "center returns both axes to startup zero");
        camera_tracking_key(manual, '2');
        camera_tracking_update(manual, {},15000);
        check(manual.mode_==TRACKING_AUTOMATIC && !manual.command_.control_valid,
              "auto without observations holds");
        CameraTracking automatic;
        for (int i=0;i<150;++i) camera_tracking_update(automatic, {{observation}},i*30.);
        check(automatic.simulated_angles_.norm()>0.1,"auto moves simulated platform");
        const auto forecast=armor_forecast(automatic.ekf_, camera_tracking_horizon_s);
        double nearest=1e9;
        for (int i=0;i<4;++i)
            nearest=std::min(nearest,(forecast.plates.row(i).head(3).transpose()-automatic.predicted_target_).norm());
        check(nearest<1e-12,"auto target is a forecast plate");
        const auto held=automatic.simulated_angles_;
        camera_tracking_update(automatic, {},4500);
        check((automatic.simulated_angles_-held).norm()==0,"loss holds simulated position");
        std::cout << "Live camera tracking checks passed\n";
    } catch(const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
