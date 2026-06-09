#include "neptunet_state_machine.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace neptunet
{

namespace
{

std::string trim(const std::string& value)
{
    const std::string whitespace = " \t\n\r\f\v";

    const std::size_t start = value.find_first_not_of(whitespace);

    if (start == std::string::npos)
    {
        return "";
    }

    const std::size_t end = value.find_last_not_of(whitespace);

    return value.substr(start, end - start + 1);
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;

    while (std::getline(stream, token, ','))
    {
        tokens.push_back(trim(token));
    }

    return tokens;
}

bool parse_bool(const std::string& value)
{
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return lowered == "1" || lowered == "true" || lowered == "yes";
}

double parse_double_or_zero(const std::string& value)
{
    if (value.empty())
    {
        return 0.0;
    }

    return std::stod(value);
}

std::optional<int> parse_optional_int(const std::string& value)
{
    if (value.empty())
    {
        return std::nullopt;
    }

    return std::stoi(value);
}

}  // namespace

NeptuNetStateMachine::NeptuNetStateMachine(double suspicion_threshold)
    : suspicion_threshold_(suspicion_threshold)
{
}

RuntimeDecision NeptuNetStateMachine::process(const InspectionWindow& window) const
{
    RuntimeDecision decision;
    decision.window_id = window.window_id;
    decision.input = window;

    // Level 1 is always active.
    decision.level1_active = true;

    // Level 2 is active only when the pipeline is detected.
    decision.level2_active = window.pipeline_detected;

    // Level 3 is active only when pipeline context exists and suspicion is high.
    decision.level3_active =
        window.pipeline_detected &&
        decision.level2_active &&
        window.suspicion_score >= suspicion_threshold_;

    if (!window.pipeline_detected)
    {
        decision.system_state = SystemState::PIPELINE_SEARCH;
    }
    else if (decision.level3_active && window.plume_detected)
    {
        decision.system_state = SystemState::LEAK_CONFIRMED;
    }
    else if (decision.level3_active && !window.plume_detected)
    {
        decision.system_state = SystemState::SUSPICION_UNCONFIRMED;
    }
    else
    {
        decision.system_state = SystemState::NORMAL_MONITORING;
    }

    return decision;
}

double NeptuNetStateMachine::suspicion_threshold() const
{
    return suspicion_threshold_;
}

std::string to_string(SystemState state)
{
    switch (state)
    {
        case SystemState::PIPELINE_SEARCH:
            return "PIPELINE_SEARCH";
        case SystemState::NORMAL_MONITORING:
            return "NORMAL_MONITORING";
        case SystemState::SUSPICION_UNCONFIRMED:
            return "SUSPICION_UNCONFIRMED";
        case SystemState::LEAK_CONFIRMED:
            return "LEAK_CONFIRMED";
        default:
            return "UNKNOWN";
    }
}

std::vector<InspectionWindow> load_scenario_csv(const std::string& csv_path)
{
    std::ifstream file(csv_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open CSV file: " + csv_path);
    }

    std::vector<InspectionWindow> scenario;
    std::string line;

    // Skip header.
    if (!std::getline(file, line))
    {
        throw std::runtime_error("CSV file is empty: " + csv_path);
    }

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        const auto tokens = split_csv_line(line);

        if (tokens.size() < 13)
        {
            throw std::runtime_error(
                "Invalid CSV row. Expected 13 fields but got " +
                std::to_string(tokens.size()) + ": " + line
            );
        }

        InspectionWindow window;

        window.window_id = std::stoi(tokens[0]);

        window.pipeline_detected = parse_bool(tokens[1]);
        window.center_offset_px = parse_double_or_zero(tokens[2]);
        window.orientation_deg = parse_double_or_zero(tokens[3]);
        window.direction = tokens[4].empty() ? "UNKNOWN" : tokens[4];

        window.bubble_activity = parse_bool(tokens[5]);
        window.suspicion_score = parse_double_or_zero(tokens[6]);

        window.plume_detected = parse_bool(tokens[7]);
        window.plume_centroid_x = parse_optional_int(tokens[8]);
        window.plume_centroid_y = parse_optional_int(tokens[9]);
        window.probable_source_x = parse_optional_int(tokens[10]);
        window.probable_source_y = parse_optional_int(tokens[11]);
        window.propagation_direction = tokens[12].empty() ? "NONE" : tokens[12];

        scenario.push_back(window);
    }

    return scenario;
}

void print_decision(const RuntimeDecision& decision)
{
    const auto& input = decision.input;

    std::cout << "[Window "
              << std::setw(2) << std::setfill('0') << decision.window_id
              << "] System State: " << to_string(decision.system_state)
              << std::setfill(' ') << "\n";

    if (decision.level1_active && input.pipeline_detected)
    {
        std::cout << "  Level 1: ACTIVE  | pipeline detected"
                  << " | center_offset=" << std::fixed << std::setprecision(2)
                  << input.center_offset_px << " px"
                  << " | orientation=" << input.orientation_deg << " deg"
                  << " | direction=" << input.direction << "\n";
    }
    else
    {
        std::cout << "  Level 1: ACTIVE  | pipeline not detected\n";
    }

    if (decision.level2_active)
    {
        std::cout << "  Level 2: ACTIVE  | bubble_activity="
                  << (input.bubble_activity ? "true" : "false")
                  << " | suspicion_score=" << std::fixed << std::setprecision(2)
                  << input.suspicion_score << "\n";
    }
    else
    {
        std::cout << "  Level 2: INACTIVE | waiting for valid pipeline context\n";
    }

    if (decision.level3_active)
    {
        std::cout << "  Level 3: ACTIVE  | plume_detected="
                  << (input.plume_detected ? "true" : "false");

        if (input.plume_detected)
        {
            std::cout << " | centroid=("
                      << input.plume_centroid_x.value_or(-1) << ","
                      << input.plume_centroid_y.value_or(-1) << ")";

            std::cout << " | source=("
                      << input.probable_source_x.value_or(-1) << ","
                      << input.probable_source_y.value_or(-1) << ")";

            std::cout << " | direction=" << input.propagation_direction;
        }

        std::cout << "\n";
    }
    else
    {
        std::cout << "  Level 3: INACTIVE | suspicion threshold not reached\n";
    }

    std::cout << "\n";
}

}  // namespace neptunet