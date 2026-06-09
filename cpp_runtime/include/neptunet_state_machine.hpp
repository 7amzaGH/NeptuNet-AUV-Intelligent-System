#ifndef NEPTUNET_STATE_MACHINE_HPP
#define NEPTUNET_STATE_MACHINE_HPP

#include <optional>
#include <string>
#include <vector>

namespace neptunet
{

enum class SystemState
{
    PIPELINE_SEARCH,
    NORMAL_MONITORING,
    SUSPICION_UNCONFIRMED,
    LEAK_CONFIRMED
};

struct InspectionWindow
{
    int window_id = 0;

    bool pipeline_detected = false;
    double center_offset_px = 0.0;
    double orientation_deg = 0.0;
    std::string direction = "UNKNOWN";

    bool bubble_activity = false;
    double suspicion_score = 0.0;

    bool plume_detected = false;
    std::optional<int> plume_centroid_x;
    std::optional<int> plume_centroid_y;
    std::optional<int> probable_source_x;
    std::optional<int> probable_source_y;
    std::string propagation_direction = "NONE";
};

struct RuntimeDecision
{
    int window_id = 0;

    bool level1_active = true;
    bool level2_active = false;
    bool level3_active = false;

    SystemState system_state = SystemState::PIPELINE_SEARCH;

    InspectionWindow input;
};

class NeptuNetStateMachine
{
public:
    explicit NeptuNetStateMachine(double suspicion_threshold = 0.70);

    RuntimeDecision process(const InspectionWindow& window) const;

    double suspicion_threshold() const;

private:
    double suspicion_threshold_;
};

std::string to_string(SystemState state);

std::vector<InspectionWindow> load_scenario_csv(const std::string& csv_path);

void print_decision(const RuntimeDecision& decision);

}  // namespace neptunet

#endif  // NEPTUNET_STATE_MACHINE_HPP