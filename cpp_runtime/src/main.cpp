#include "neptunet_state_machine.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <scenario_csv> [suspicion_threshold]\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " ../examples/scenario_04_confirmed_leak.csv\n"
                  << "  " << argv[0] << " ../examples/scenario_04_confirmed_leak.csv 0.75\n";

        return EXIT_FAILURE;
    }

    const std::string scenario_path = argv[1];

    double suspicion_threshold = 0.70;

    if (argc >= 3)
    {
        suspicion_threshold = std::stod(argv[2]);
    }

    try
    {
        const auto scenario = neptunet::load_scenario_csv(scenario_path);

        neptunet::NeptuNetStateMachine state_machine(suspicion_threshold);

        std::cout << "NeptuNet C++ Embedded Runtime Skeleton\n";
        std::cout << "Scenario file: " << scenario_path << "\n";
        std::cout << "Suspicion threshold: "
                  << state_machine.suspicion_threshold() << "\n\n";

        for (const auto& window : scenario)
        {
            const auto decision = state_machine.process(window);
            neptunet::print_decision(decision);
        }

        std::cout << "Simulation completed.\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "Runtime error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}