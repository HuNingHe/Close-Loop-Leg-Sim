/*!
 * @file:                              draw_foot_trail.cpp
 * @date:                                  2022.07.12
 * @author:                                 HuNing-He
 * @email:                               huning-he@qq.com
 * @description:                          draw foot trail
 */
#include <webots/Field.hpp>
#include <webots/Node.hpp>
#include <webots/Supervisor.hpp>
#include <webots/Receiver.hpp>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

using webots::Supervisor;
using webots::Receiver;
using webots::Node;
using webots::Field;

struct Foot_Pos{
    double foot_pos[3];
};

const unsigned int trail_fresh_factor = 10;           // Trajectory update speed, updated every 10 time_step, reduces CPU and GPU usage
const unsigned int max_num_of_trail_coordinates = 100; // Trajectory update speed, updated every 10 time_step, reduces CPU and GPU usage

void create_trail(std::shared_ptr<Supervisor> sup, const std::string& target, std::vector<double> &color) {
    if (color.size() < 3) {
        throw std::runtime_error("color vector size error");
    }
    // if this trail have existed, delete it first
    Node *existing_trail = sup->getFromDef(target);
    if (existing_trail)
        existing_trail->remove();
    for (int i = 0; i < 3; ++i) {
        if (color[i] > 1){
            color[i] = 1;
        } else if (color[i] <= 0) {
            color[i] = 0;
        }
    }

    std::string r = std::to_string(color[0]);
    std::string g = std::to_string(color[1]);
    std::string b = std::to_string(color[2]);

    std::string trail_string;
    trail_string.clear();
    trail_string.append("DEF ");
    trail_string.append(target);
    trail_string.append(" Shape {\n  appearance Appearance {\n    material Material {\n");
    trail_string.append("      diffuseColor ");

    trail_string.append(r);
    trail_string.append(" ");
    trail_string.append(g);
    trail_string.append(" ");
    trail_string.append(b);
    trail_string.append("\n");

    trail_string.append("      emissiveColor ");

    trail_string.append(r);
    trail_string.append(" ");
    trail_string.append(g);
    trail_string.append(" ");
    trail_string.append(b);
    trail_string.append("\n");

    trail_string.append("    }\n  }\n  geometry DEF ");
    trail_string.append(target);
    trail_string.append("_LINE_SET"); // Note that all target trajectories will be followed by a "_LINE_SET" to avoid DEF conflicts

    trail_string.append(" IndexedLineSet {\n    coord Coordinate {\n      point [\n");
    for (int i = 0; i < max_num_of_trail_coordinates; ++i)
        trail_string.append("      0 0 0\n");
    trail_string.append("      ]\n    }\n    coordIndex [\n");

    for (int i = 0; i < max_num_of_trail_coordinates; ++i)
        trail_string.append("      0 0 -1\n");
    trail_string.append("    ]\n  }\n}\n");
    // Create a track under the world node
    Node *root_node = sup->getRoot();
    Field *root_children_field = root_node->getField("children");
    root_children_field->importMFNodeFromString(-1, trail_string);
}

int main() {
    auto super_robot = std::make_shared<Supervisor>();
    auto receiver = super_robot->getReceiver("receiver");
    int time_step = static_cast<int>(super_robot->getBasicTimeStep());

    receiver->enable(time_step);
    receiver->setChannel(100);

    std::vector<double> act_trail_color = {1, 0, 0};
    std::vector<double> des_trail_color = {0, 0, 1};

    Node *foot_node = super_robot->getFromDef("Foot");
    Node *hip_node = super_robot->getFromDef("Hip");

    std::string target_def = "Foot_Pos";
    std::string des_target_def = "Des_Foot_Pos";
    create_trail(super_robot, target_def, act_trail_color); // actual foot pos trail node
    create_trail(super_robot, des_target_def, des_trail_color); // des foot pos trail node

    Node *trail_node = super_robot->getFromDef("Foot_Pos_LINE_SET");
    Node *des_trail_node = super_robot->getFromDef("Des_Foot_Pos_LINE_SET");

    Node *coord_node = trail_node->getField("coord")->getSFNode();
    Node *des_coord_node = des_trail_node->getField("coord")->getSFNode();

    Field *trail_point_field = coord_node->getField("point");
    Field *des_trail_point_field = des_coord_node->getField("point");

    Field *trail_coord_index_field = trail_node->getField("coordIndex");
    Field *des_trail_coord_index_field = des_trail_node->getField("coordIndex");

    int index = 0;
    unsigned long int iter = 0;
    bool first_step = true;

    double des_foot_pos[3] = {0, 0, 0};
    const struct Foot_Pos *des_pos;

    while (super_robot->step(time_step) != -1) {
        const double *value = foot_node->getPosition();
        const double *hip_pos = hip_node->getPosition();

//        std::cout << "actual foot pos: \n" << value[0] << " " << value[1] << " " << value[2] << std::endl;
        if (receiver->getQueueLength() > 0) {
            des_pos = reinterpret_cast<const Foot_Pos*>(receiver->getData());
            des_foot_pos[0] = des_pos->foot_pos[0] + hip_pos[0];
            des_foot_pos[1] = des_pos->foot_pos[1] + hip_pos[1];
            des_foot_pos[2] = des_pos->foot_pos[2] + hip_pos[2];
        } else {
            std::cout << " No Message Received !" << std::endl;
        }
        // The following sentence cannot be omitted!!! Otherwise, the first frame is read forever
        receiver->nextPacket();

        iter++;
        const double *target_translation = foot_node->getPosition();

        if (iter % trail_fresh_factor == 0) {
            trail_point_field->setMFVec3f(index, target_translation);
            des_trail_point_field->setMFVec3f(index, des_foot_pos);

            if (index > 0) {
                trail_coord_index_field->setMFInt32(3 * (index - 1), index - 1);
                trail_coord_index_field->setMFInt32(3 * (index - 1) + 1, index);

                des_trail_coord_index_field->setMFInt32(3 * (index - 1), index - 1);
                des_trail_coord_index_field->setMFInt32(3 * (index - 1) + 1, index);
            } else if(index == 0 && !first_step) {
                trail_coord_index_field->setMFInt32(3 * (max_num_of_trail_coordinates - 1), 0);
                trail_coord_index_field->setMFInt32(3 * (max_num_of_trail_coordinates - 1) + 1, max_num_of_trail_coordinates - 1);

                des_trail_coord_index_field->setMFInt32(3 * (max_num_of_trail_coordinates - 1), 0);
                des_trail_coord_index_field->setMFInt32(3 * (max_num_of_trail_coordinates - 1) + 1, max_num_of_trail_coordinates - 1);
            }
            trail_coord_index_field->setMFInt32(3 * index, index);
            trail_coord_index_field->setMFInt32(3 * index + 1, index);

            des_trail_coord_index_field->setMFInt32(3 * index, index);
            des_trail_coord_index_field->setMFInt32(3 * index + 1, index);

            first_step = false;
            index++;
            index %= max_num_of_trail_coordinates;
        }
    }
    return 0;
}
