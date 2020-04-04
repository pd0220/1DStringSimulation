// 1D elastic string simulation (solving the wave equation numerically)

// including useful headers and/or libraries
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
// Eigen library for matrices
#include <Eigen/Dense>

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// number of time and space points on the lattice
const int numOfTimeSteps = 10000;
const int numOfSpaceSteps = 200;
// speed of wave [sqrt(tension / linear density)]
const double cWave = 10.;
// starting and ending (x = L) points in space
const double xStart = 0., xStop = 1.;
// starting and ending points in time
const double tStart = 0., tStop = 2.;
// step size for time and space
const double deltaT = (tStop - tStart) / numOfTimeSteps;
const double deltaX = (xStop - xStart) / numOfSpaceSteps;
// characteristic constant for stepping
const double StepperConst = cWave * cWave * deltaT * deltaT / deltaX / deltaX;
// initial conditions --> x0 and y0
const double X0 = xStop / 2;
const double Y0 = 0.01;
// file name
std::string fileName = "data_mixed_beat.txt";

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// setting initial value for beat pattern interference
// hard coded: sum of two normal modes
auto SetInitialValueBeat = [&](Eigen::MatrixXd &mat, int mode1, int mode2)
{
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // set initial value
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
    {
        double x = xIndex * deltaX;
        mat(0, xIndex) = Y0 * (std::sin(mode1 * M_PI / (xStop - deltaX) * x) + std::sin(mode2 * M_PI / (xStop - deltaX) * x));
    }


    // y(x = 0, t) = y(x = L, t) = 0
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        mat(tIndex, 0) = 0.;
        mat(tIndex, numOfSpaceSteps - 1) = 0.;
    }
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// setting initial values for linearity testing
// left, right or both
auto SetInitialValueLinear = [&](Eigen::MatrixXd &mat, std::string detSwitch) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // left
    if (detSwitch == "left")
    {
        double inRatio = Y0 / xStop * 3;
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= (double)xStop / 3)
            {
                mat(0, xIndex) = inRatio * x;
            }
            else
            {
                mat(0, xIndex) = -inRatio * (x - xStop + deltaX) / 2;
            }
        }
    }
    // right
    else if (detSwitch == "right")
    {
        double inRatio = Y0 / xStop * 3 / 2;
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= (double)2 / 3 * xStop)
            {
                mat(0, xIndex) = -inRatio * x;
            }
            else
            {
                mat(0, xIndex) = inRatio * (x - xStop + deltaX) * 2;
            }
        }
    }

    // both
    if (detSwitch == "both")
    {
        double inRatio = Y0 / xStop * 3 / 2;
        // second mode
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= (double)xStop / 3)
            {
                mat(0, xIndex) = inRatio * x;
            }
            else if (x > (double)xStop / 3 && x <= (double)xStop * 2 / 3)
            {
                mat(0, xIndex) = -inRatio * (x - (xStop - deltaX) / 2) * 2;
            }
            else
            {
                mat(0, xIndex) = inRatio * (x - xStop + deltaX);
            }
        }
    }

    // y(x = 0, t) = y(x = L, t) = 0
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        mat(tIndex, 0) = 0.;
        mat(tIndex, numOfSpaceSteps - 1) = 0.;
    }
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// setting initial values (lambda) --> strumming
// X0 must be L / mode
auto SetInitialValueStrum = [&](Eigen::MatrixXd &mat, int mode) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }
    // check space coordinate of initial value
    if (X0 <= xStart || X0 >= xStop)
    {
        std::cout << "ERROR\nGive space coordinate is not appropriate." << std::endl;
        std::exit(-1);
    }
    // check mode
    if (mode != 1 && mode != 2 && mode != 3)
    {
        std::cout << "ERROR\nGiven order is not available." << std::endl;
        std::exit(-1);
    }

    double inRatio = Y0 / X0;

    // y(x, t = 0) = given value
    if (mode == 1)
    {
        // first mode
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= X0)
            {
                mat(0, xIndex) = inRatio * x;
            }
            else
            {
                mat(0, xIndex) = -inRatio * (x - xStop);
            }
        }
    }
    else if (mode == 2)
    {
        // second mode
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= X0)
            {
                mat(0, xIndex) = inRatio * x;
            }
            else if (x > X0 && x <= 3 * X0)
            {
                mat(0, xIndex) = -inRatio * (x - xStop / 2);
            }
            else if (x > 3 * X0)
            {
                mat(0, xIndex) = inRatio * (x - xStop);
            }
        }
    }
    else if (mode == 3)
    {
        // third mode
        for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        {
            double x = xIndex * deltaX;
            if (x <= X0)
            {
                mat(0, xIndex) = inRatio * x;
            }
            else if (x > X0 && x <= 3 * X0)
            {
                mat(0, xIndex) = -inRatio * (x - xStop / 3);
            }
            else if (x > 3 * X0 && x <= 5 * X0)
            {
                mat(0, xIndex) = inRatio * (x - 2 * xStop / 3);
            }
            else if (x > 5 * X0)
            {
                mat(0, xIndex) = -inRatio * (x - xStop);
            }
        }
    }

    // y(x = 0, t) = y(x = L, t) = 0
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        mat(tIndex, 0) = 0.;
        mat(tIndex, numOfSpaceSteps - 1) = 0.;
    }
};

// setting initial values as normal modes
auto SetInitialValue = [&](Eigen::MatrixXd &mat, int mode) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }
    // check mode
    if (mode != 1 && mode != 2 && mode != 3)
    {
        std::cout << "ERROR\nGiven order is not available." << std::endl;
        std::exit(-1);
    }

    // set normal mode
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
    {
        double x = xIndex * deltaX;
        mat(0, xIndex) = Y0 * std::sin(mode * M_PI / (xStop - deltaX) * x);
    }

    // y(x = 0, t) = y(x = L, t) = 0
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        mat(tIndex, 0) = 0.;
        mat(tIndex, numOfSpaceSteps - 1) = 0.;
    }
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// taking first step (lambda)
auto FirstStep = [](Eigen::MatrixXd &mat) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // initial velocity will be dy(x, t = 0) / dt = f(x) = 0 --> taking first step accordingly
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        mat(1, xIndex) = mat(0, xIndex) * (1 - StepperConst) + StepperConst / 2 * (mat(0, xIndex + 1) + mat(0, xIndex - 1));
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// taking a single step (from tIndex to tIndex + 1)
auto Step = [](Eigen::MatrixXd &mat, int tIndex) {
    // check matrix dimensions
    if (mat.rows() != numOfTimeSteps || mat.cols() != numOfSpaceSteps)
    {
        std::cout << "ERROR\nGiven matrix dimensions are not appropriate." << std::endl;
        std::exit(-1);
    }

    // taking one timestep
    for (int xIndex{1}; xIndex < numOfSpaceSteps - 1; xIndex++)
        mat(tIndex + 1, xIndex) = 2 * mat(tIndex, xIndex) * (1 - StepperConst) - mat(tIndex - 1, xIndex) + StepperConst * (mat(tIndex, xIndex + 1) + mat(tIndex, xIndex - 1));
};

// -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// main function
int main(int, char **)
{
    // check for stability
    if (StepperConst >= 1)
    {
        std::cout << "WARNING\nStability issues may occur." << std::endl;
        std::cout << "StepperConst = " << StepperConst << std::endl;
    }

    // matrix for y(x, t) values
    Eigen::MatrixXd matSolution(numOfTimeSteps, numOfSpaceSteps);

    // setting initial values for y(x = 0, t) = y(x = L, t) = 0 and y(x, t = 0) --> normal modes
    //SetInitialValue(matSolution, 1);
    //SetInitialValueStrum(matSolution, 1);
    //SetInitialValueLinear(matSolution, "both");
    SetInitialValueBeat(matSolution, 20, 21);

    // taking the first step seperately --> estimating y(x, t - dt) from initial condition: dy(x, t = 0) / dt = f(x)
    FirstStep(matSolution);

    // taking timesteps (rest of the simulation)
    for (int tIndex{1}; tIndex < numOfTimeSteps - 1; tIndex++)
    {
        Step(matSolution, tIndex);
    }

    // simulation data
    std::ofstream data;
    data.open("animationData.txt");
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        for (int xIndex{0}; xIndex < numOfSpaceSteps; xIndex++)
        {
            data << matSolution(tIndex, xIndex) << " ";
        }
        data << "\n";
    }
    data.close();

    // analysis data
    data.open(fileName);
    for (int tIndex{0}; tIndex < numOfTimeSteps; tIndex++)
    {
        for (int xIndex{0}; xIndex < numOfSpaceSteps; xIndex++)
        {
            data << tIndex * deltaT << " " << xIndex * deltaX << " " << matSolution(tIndex, xIndex) << "\n";
        }
    }
    data.close();

    std::cout << deltaX / deltaT << std::endl;
}
