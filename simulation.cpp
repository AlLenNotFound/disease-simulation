#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// Model parameters
double beta = 0.5;
double gammaRate = 0.1;
double dt = 0.01;
int numSteps = 1000;

struct SIR {
    double S, I, R;
};

// Reads a portion of the CSV file in parallel
std::vector<std::vector<SIR>> loadLocalData(const std::string& filename, int startRow, int localRows, int ncols) {
    std::vector<std::vector<SIR>> grid(localRows, std::vector<SIR>(ncols));
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string line;
    for (int i = 0; i < startRow + localRows; ++i) {
        std::getline(infile, line);
        if (i >= startRow) {
            std::istringstream ss(line);
            int r, c;
            double S, I, R;
            ss >> r >> c >> S >> I >> R;
            grid[r - startRow][c] = {S, I, R};
        }
    }
    return grid;
}

SIR rk4Step(const SIR &current) {
    auto fS = [&](const SIR &state) { return -beta * state.S * state.I; };
    auto fI = [&](const SIR &state) { return beta * state.S * state.I - gammaRate * state.I; };
    auto fR = [&](const SIR &state) { return gammaRate * state.I; };
    
    SIR k1, k2, k3, k4, next, temp;
    k1 = {dt * fS(current), dt * fI(current), dt * fR(current)};
    temp = {current.S + 0.5 * k1.S, current.I + 0.5 * k1.I, current.R + 0.5 * k1.R};
    k2 = {dt * fS(temp), dt * fI(temp), dt * fR(temp)};
    temp = {current.S + 0.5 * k2.S, current.I + 0.5 * k2.I, current.R + 0.5 * k2.R};
    k3 = {dt * fS(temp), dt * fI(temp), dt * fR(temp)};
    temp = {current.S + k3.S, current.I + k3.I, current.R + k3.R};
    k4 = {dt * fS(temp), dt * fI(temp), dt * fR(temp)};
    next = {current.S + (k1.S + 2*k2.S + 2*k3.S + k4.S) / 6.0,
            current.I + (k1.I + 2*k2.I + 2*k3.I + k4.I) / 6.0,
            current.R + (k1.R + 2*k2.R + 2*k3.R + k4.R) / 6.0};
    return next;
}

void updateGrid(std::vector<std::vector<SIR>> &grid) {
    std::vector<std::vector<SIR>> newGrid = grid;
    for (size_t i = 0; i < grid.size(); ++i) {
        for (size_t j = 0; j < grid[0].size(); ++j) {
            newGrid[i][j] = rk4Step(grid[i][j]);
        }
    }
    grid = newGrid;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int nrows = 100, ncols = 100; // Assume fixed size or read from file
    int rowsPerProc = nrows / size;
    int startRow = rank * rowsPerProc;
    
    auto localGrid = loadLocalData("initial_conditions.csv", startRow, rowsPerProc, ncols);
    
    for (int step = 0; step < numSteps; ++step) {
        updateGrid(localGrid);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    std::vector<SIR> flattened(rowsPerProc * ncols);
    for (int i = 0; i < rowsPerProc; ++i)
        for (int j = 0; j < ncols; ++j)
            flattened[i * ncols + j] = localGrid[i][j];
    
    if (rank == 0) {
        std::vector<SIR> globalData(nrows * ncols);
        MPI_Gather(flattened.data(), rowsPerProc * ncols * 3, MPI_DOUBLE,
                   globalData.data(), rowsPerProc * ncols * 3, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        std::ofstream outfile("simulation_results.csv");
        outfile << "row,col,S,I,R\n";
        for (int i = 0; i < nrows; ++i)
            for (int j = 0; j < ncols; ++j)
                outfile << i << "," << j << ","
                        << globalData[i * ncols + j].S << ","
                        << globalData[i * ncols + j].I << ","
                        << globalData[i * ncols + j].R << "\n";
        outfile.close();
    } else {
        MPI_Gather(flattened.data(), rowsPerProc * ncols * 3, MPI_DOUBLE,
                   nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}