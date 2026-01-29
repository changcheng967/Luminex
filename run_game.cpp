// Simple UCI game player for self-play
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

string exec_command(const string& cmd) {
    char buffer[128];
    string result;
    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    _pclose(pipe);
    return result;
}

int main() {
    string pos = "startpos";
    vector<string> moves;
    int move_count = 0;
    int max_moves = 100;

    cout << "=== Luminex v2.2.0 Self-Play ===" << endl;
    cout << "Time control: 500ms per move" << endl;
    cout << "================================" << endl << endl;

    while (move_count < max_moves) {
        // White move
        string cmd = "cd C:/Users/chang/Downloads/Luminex/build && echo \"position " + pos +
                     "\" && echo \"go movetime 500\" && echo \"quit\" | ./luminex.exe 2>&1";
        string output = exec_command(cmd);

        // Find bestmove
        size_t best_pos = output.find("bestmove ");
        if (best_pos != string::npos) {
            string move = output.substr(best_pos + 9, 4);
            cout << (move_count/2 + 1) << ". " << move << " ";
            moves.push_back(move);

            if (move == "(none)" || move == "0000") {
                cout << "\nGame over - no legal moves" << endl;
                break;
            }

            // Update position
            if (pos == "startpos") {
                pos = "startpos moves " + move;
            } else {
                pos = pos + " " + move;
            }
        } else {
            cout << "White: No move found" << endl;
            break;
        }

        move_count++;
        if (move_count >= max_moves) break;

        // Black move
        cmd = "cd C:/Users/chang/Downloads/Luminex/build && echo \"position " + pos +
              "\" && echo \"go movetime 500\" && echo \"quit\" | ./luminex.exe 2>&1";
        output = exec_command(cmd);

        best_pos = output.find("bestmove ");
        if (best_pos != string::npos) {
            string move = output.substr(best_pos + 9, 4);
            cout << move << " ";
            moves.push_back(move);

            if (move == "(none)" || move == "0000") {
                cout << "\nGame over - no legal moves" << endl;
                break;
            }

            pos = pos + " " + move;

            // Print eval
            size_t score_pos = output.find("score cp ");
            if (score_pos != string::npos) {
                size_t space_pos = output.find(" ", score_pos + 10);
                if (space_pos != string::npos) {
                    string score = output.substr(score_pos + 10, space_pos - score_pos - 10);
                    cout << "[eval: " << score << "]";
                }
            }
            cout << endl;
        } else {
            cout << "Black: No move found" << endl;
            break;
        }

        move_count++;
    }

    cout << "\n=== Final Position ===" << endl;
    cout << "Moves: " << move_count << endl;
    cout << "PGN: ";
    for (size_t i = 0; i < moves.size(); i += 2) {
        cout << (i/2 + 1) << ". " << moves[i] << " ";
        if (i + 1 < moves.size()) {
            cout << moves[i + 1] << " ";
        }
    }
    cout << endl;

    return 0;
}
