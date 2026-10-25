#include <iostream>
#include <fstream>
#include <iomanip>
#include "/tmp/hldemo_cpp/src/DemoFile.hpp"

// Button definitions from Half-Life SDK
#define IN_ATTACK   (1 << 0)
#define IN_JUMP     (1 << 1)
#define IN_DUCK     (1 << 2)
#define IN_FORWARD  (1 << 3)
#define IN_BACK     (1 << 4)
#define IN_USE      (1 << 5)
#define IN_MOVELEFT (1 << 9)
#define IN_MOVERIGHT (1 << 10)
#define IN_ATTACK2  (1 << 11)
#define IN_RELOAD   (1 << 13)

int main(int argc, char* argv[]) {
    std::string demo_path = "dataset/session_2000.dem";
    std::string output_path = "dataset/extracted_actions.jsonl";

    if (argc > 1) demo_path = argv[1];
    if (argc > 2) output_path = argv[2];

    std::cout << "[Extractor] Opening demo: " << demo_path << std::endl;
    DemoFile demo(demo_path, true);

    std::cout << "[Extractor] Map: " << demo.header.mapName << std::endl;
    std::cout << "[Extractor] Tracks count: " << demo.directoryEntries.size() << std::endl;

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return 1;
    }

    size_t total_samples = 0;
    size_t attack_count = 0;
    size_t jump_count = 0;

    for (size_t track_idx = 0; track_idx < demo.directoryEntries.size(); ++track_idx) {
        const auto& entry = demo.directoryEntries[track_idx];
        std::cout << "[Extractor] Processing Track " << track_idx 
                  << " (\"" << entry.description << "\"), frames: " 
                  << entry.frames.size() << std::endl;

        float prev_pitch = 0.0f;
        float prev_yaw = 0.0f;
        bool has_prev = false;

        for (const auto& f : entry.frames) {
            // NetMsgFrame contains user actions and coordinates
            auto netmsg = dynamic_cast<const NetMsgFrame*>(f.get());
            if (!netmsg) continue;

            const auto& ucmd = netmsg->DemoInfo.UserCmd;
            const auto& ref = netmsg->DemoInfo.RefParams;

            float pitch = ucmd.viewangles[0];
            float yaw = ucmd.viewangles[1];

            float delta_pitch = 0.0f;
            float delta_yaw = 0.0f;

            if (has_prev) {
                delta_pitch = pitch - prev_pitch;
                delta_yaw = yaw - prev_yaw;
                // Handle 360 wrap-around
                if (delta_yaw > 180.0f) delta_yaw -= 360.0f;
                if (delta_yaw < -180.0f) delta_yaw += 360.0f;
            } else {
                has_prev = true;
            }
            prev_pitch = pitch;
            prev_yaw = yaw;

            uint16_t btn = ucmd.buttons;
            int attack = (btn & IN_ATTACK) ? 1 : 0;
            int attack2 = (btn & IN_ATTACK2) ? 1 : 0;
            int jump = (btn & IN_JUMP) ? 1 : 0;
            int duck = (btn & IN_DUCK) ? 1 : 0;
            int reload = (btn & IN_RELOAD) ? 1 : 0;

            int move_w = (ucmd.forwardmove > 50.0f || (btn & IN_FORWARD)) ? 1 : 0;
            int move_s = (ucmd.forwardmove < -50.0f || (btn & IN_BACK)) ? 1 : 0;
            int move_a = (ucmd.sidemove < -50.0f || (btn & IN_MOVELEFT)) ? 1 : 0;
            int move_d = (ucmd.sidemove > 50.0f || (btn & IN_MOVERIGHT)) ? 1 : 0;

            if (attack) attack_count++;
            if (jump) jump_count++;

            out << "{"
                << "\"frame\":" << f->frame << ","
                << "\"time\":" << std::fixed << std::setprecision(3) << f->time << ","
                << "\"health\":" << ref.health << ","
                << "\"origin\":[" << ref.simorg[0] << "," << ref.simorg[1] << "," << ref.simorg[2] << "],"
                << "\"angles\":[" << pitch << "," << yaw << "],"
                << "\"delta_angles\":[" << delta_pitch << "," << delta_yaw << "],"
                << "\"keys\":{"
                << "\"w\":" << move_w << ",\"s\":" << move_s << ","
                << "\"a\":" << move_a << ",\"d\":" << move_d << ","
                << "\"space\":" << jump << ",\"ctrl\":" << duck << ","
                << "\"r\":" << reload
                << "},"
                << "\"mouse\":{"
                << "\"dx\":" << delta_yaw << ",\"dy\":" << delta_pitch << ","
                << "\"left\":" << attack << ",\"right\":" << attack2
                << "}"
                << "}\n";

            total_samples++;
        }
    }

    out.close();
    std::cout << "\n[Extractor DONE] Successfully extracted " << total_samples << " ticks!" << std::endl;
    std::cout << "  - Attack events: " << attack_count << std::endl;
    std::cout << "  - Jump events: " << jump_count << std::endl;
    std::cout << "  - Saved to: " << output_path << std::endl;
    return 0;
}
