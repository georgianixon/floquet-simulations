#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <list>
#include <algorithm> // std::count
#include <set>
#include "BasicFunctions.h"
#include "Decoder.h"
#include "CreateEdgeWeightLookup.h"
#include "ImportN.h"


using namespace std;


struct BoolInt {
	bool A;
	int B;
};

struct IntInt {
	int A;
	int B;
};


BoolInt TestCode(float p, int L, int H, int MinTries, vector<int> Symmetry, int SymmetrySize, vector<vector<int> > MatrixEdgeWeights )
{

	
	// * Initalise lattice *
	vector<int> Qubits(L*H, 0);     // L*H qubits (one on each vertex)
	vector<int> Stabilizers(L*H, 0);  // L*H stabilizer measurement outcomes (one on each placet) (dual lattice of qubits)

	string Strategy = "Flexible";

	// Add an error to each qubit with probability 'p', initialised above
	//cout << "\nErrors:\n";
	for (int j = 0; j < L*H; j++)
	{
		if (rand() / (float)RAND_MAX < p)
		{
			Qubits[j] = 1;
			//cout << j << ", ";
		}
	}
	
	//vector<int> ErrorLocations{ };
	//for (auto const& i : ErrorLocations) {
	//	Qubits[i] = 1;
	//}

	for (int j = 0; j < L * H; j++)
	{
		// Function 'StarMeasurement' evaluates star measurements with periodic boundary conditions
		Stabilizers[j] = StarMeasurement(Qubits, j, L, H);
	}
	
	//set<tuple <int, int> > EmptySetTup;
	//vector<int> EmptyVec(L*H);
	//string title = "InitialError.asy";
	//AsymptotePrint(Stabilizers, Qubits, EmptySetTup, L, H, title);
	
	int midVertex = L * (H - 1) + L / 2;
	int bottomVertex = L * (H - 1);

	//Store results uniquely in
	vector<int> ErrorPrediction(L*H, 0);

	// Do a couple of times
	bool GiveUp = 0;
	int RoundNumber = 0;

	// the error we are actually keeping track of
	vector<int> StabilizersLeft = Stabilizers;

	while(!GiveUp)
	{
		// Find number of stabilizers from previous round
		int NumberOfOldStabilizers = 0;
		for (int j = 0; j < L*H; ++j)
		{
			if (StabilizersLeft[j])
			{
				++NumberOfOldStabilizers;
			}
		}


		// ** Initialization for Strategy = fixed
		// New flips from this round
		vector<int> NewErrorPrediction(L*H, 0);
		// Keep track of constraints
		int Constraints = 0;
		/*if (PositiveModulo(RoundNumber, 2) == 0)
		{
			Constraints = 1; // Vertical
		}
		else if (PositiveModulo(RoundNumber, 2) == 1)
		{
			Constraints = 2; // Horizontal
		}*/

		// Need for both
		vector<int> ShouldWeFlip(L*H, 0);
		
		// ** Initialization for Strategy = flexible
		// New flips from this round

		vector<int> NewErrorPredictionVertical(L*H, 0);
		vector<int> NewErrorPredictionHorizontal(L*H, 0);
		vector<int> NewErrorPredictionBoth(L*H, 0);

		// Total flips from this round
		vector<int> ErrorPredictionVertical(L*H, 0);
		vector<int> ErrorPredictionHorizontal(L*H, 0);
		vector<int> ErrorPredictionBoth(L*H, 0);
		ErrorPredictionVertical = ErrorPrediction;
		ErrorPredictionHorizontal = ErrorPrediction;
		ErrorPredictionBoth = ErrorPrediction;
		


		//Traverse through possible starting points for symmetry
		for (int j = 0; j < L*H; ++j)
		{
			// move the symmetry along
			//set<tuple<int, int> > PairSet;
			if (j != 0)
			{
				for (vector<int>::iterator d = Symmetry.begin(); d != Symmetry.end(); ++d)
					*d = EastVertex(*d, L);

				midVertex = EastVertex(midVertex, L);
				bottomVertex = EastVertex(bottomVertex, L);

				if (j % L == 0)
				{
					for (vector<int>::iterator d = Symmetry.begin(); d != Symmetry.end(); ++d)
						*d = SouthVertex(*d, L, H);

					midVertex = SouthVertex(midVertex, L, H);
					bottomVertex = SouthVertex(bottomVertex, L, H);
				}
			}
			 
			
			vector<int> DefectLocations = FindCorrespondingIndicies(Symmetry, StabilizersLeft, L*H);
			vector<int> LatticeIndexedBySymmetryNum = CreateSymmetryReverse(Symmetry, L*H);
			vector<int> DefectSymmetryLocations = IndexDefectLocationsBySymmetryNum(LatticeIndexedBySymmetryNum, DefectLocations);

			TwoLists Results = MinimumWeightMatchingDecoderFibbonacciUsingSymmetryIndex(DefectSymmetryLocations, MatrixEdgeWeights, SymmetrySize, L, H);

			for (int i = 0; i < Results.A.size(); ++i)
			{
				Results.A[i] = Symmetry[Results.A[i]];
			}
			for (vector<int>::iterator entry = Results.B.begin(); entry != Results.B.end(); ++entry)
			{
				*entry = Symmetry[*entry];
			}

			int column = VertexXPosition(midVertex, L);
			int row = VertexYPosition(bottomVertex, L);

			vector<int>::iterator ItA = Results.A.begin();
			vector<int>::iterator ItB = Results.B.begin();
			
			int NumberOfVerticalLineCrossings = 0;
			int NumberOfHorizontalLineCrossings = 0;
			
			while (ItA != Results.A.end() || ItB != Results.B.end())
			{
				// figure out if each matching crosses the line:
				set<int> XCover = FindXCoverFromPair(*ItA, *ItB, L, H);
				set<int> YCover = FindYCoverFromPair(*ItA, *ItB, L, H);

				if (XCover.find(column) != XCover.end())
					++NumberOfVerticalLineCrossings;

				if (YCover.find(SouthRow(row, H)) != YCover.end() && YCover.find(row) != YCover.end())
					++NumberOfHorizontalLineCrossings;

				//tuple <int, int> tup1(*ItA, *ItB);
				//PairSet.insert(tup1);
				++ItA;
				++ItB;
			}
			

			if (Strategy == "Fixed")
			{
				// Correct if only one line crossing each line 
				if (NumberOfVerticalLineCrossings % 2 == 1)
				{
					//only correct if we are interested in horizontal line requirements
					if (Constraints == 1)
					{
						ErrorPrediction[midVertex] = switchbool(ErrorPrediction[midVertex]);
						NewErrorPrediction[midVertex] = switchbool(NewErrorPrediction[midVertex]);
					}
					// if doing both, need to keep track and flip at the end
					else if (Constraints == 0)
						ShouldWeFlip[midVertex] += 1;
				}
				if (NumberOfHorizontalLineCrossings % 2 == 1)
				{
					//only correct if we are interested in vertical line requirements
					if (Constraints == 2)
					{
						ErrorPrediction[bottomVertex] = switchbool(ErrorPrediction[bottomVertex]);
						NewErrorPrediction[bottomVertex] = switchbool(NewErrorPrediction[bottomVertex]);
					}
					// if doing both, need to keep track and flip at the end
					else if (Constraints == 0)
						ShouldWeFlip[bottomVertex] += 1;
				}
			}
			
			
			
			if (Strategy == "Flexible")
			{
				if (NumberOfVerticalLineCrossings % 2 == 1)
				{
					NewErrorPredictionVertical[midVertex] = switchbool(NewErrorPredictionVertical[midVertex]);
					ErrorPredictionVertical[midVertex] = switchbool(ErrorPredictionVertical[midVertex]);
					ShouldWeFlip[midVertex] += 1;
				}
				if (NumberOfHorizontalLineCrossings % 2 == 1)
				{
					NewErrorPredictionHorizontal[bottomVertex] = switchbool(NewErrorPredictionHorizontal[bottomVertex]);
					ErrorPredictionHorizontal[bottomVertex] = switchbool(ErrorPredictionHorizontal[bottomVertex]);
					ShouldWeFlip[bottomVertex] += 1;
				}
			}
			
		}
		
		if (Strategy == "Fixed")
		{
			// if we need both constraints, now join all symmetry results to find which qubits satisfied both vertical and horizonal requirements
			if (Constraints == 0)
			{
				for (int j = 0; j < L*H; ++j)
				{
					if (ShouldWeFlip[j] == 2) // both horizontal and vertical satisfied
					{
						ErrorPrediction[j] = switchbool(ErrorPrediction[j]);
						NewErrorPrediction[j] = switchbool(NewErrorPrediction[j]);
					}
				}
			}

			for (int j = 0; j < L*H; ++j)
			{
				if (NewErrorPrediction[j])
				{
					//flip stabilizers
					int vertex1 = WestVertex(j, L);
					int vertex2 = EastVertex(j, L);
					int vertex3 = SouthVertex(j, L, H);

					StabilizersLeft[j] = switchbool(StabilizersLeft[j]);
					StabilizersLeft[vertex1] = switchbool(StabilizersLeft[vertex1]);
					StabilizersLeft[vertex2] = switchbool(StabilizersLeft[vertex2]);
					StabilizersLeft[vertex3] = switchbool(StabilizersLeft[vertex3]);
				}
			}
		}

		if (Strategy == "Flexible")
		{
			// Take care of the Both requirements
			for (int j = 0; j < L*H; ++j)
			{
				if (ShouldWeFlip[j] == 2) // both horizontal and vertical satisfied
				{
					ErrorPredictionBoth[j] = switchbool(ErrorPredictionBoth[j]);
					NewErrorPredictionBoth[j] = switchbool(NewErrorPredictionBoth[j]);
				}
			}

			// initialise stabilizersLeft for all options
			vector<int> StabilizersLeftTryVertical = StabilizersLeft;
			vector<int> StabilizersLeftTryHorizontal = StabilizersLeft;
			vector<int> StabilizersLeftTryBoth = StabilizersLeft;


			// flip the stabilizers we predicted from this round
			for (int j = 0; j < L*H; ++j)
			{
				if (NewErrorPredictionVertical[j])
				{
					//flip stabilizers
					int vertex1 = WestVertex(j, L);
					int vertex2 = EastVertex(j, L);
					int vertex3 = SouthVertex(j, L, H);

					StabilizersLeftTryVertical[j] = switchbool(StabilizersLeftTryVertical[j]);
					StabilizersLeftTryVertical[vertex1] = switchbool(StabilizersLeftTryVertical[vertex1]);
					StabilizersLeftTryVertical[vertex2] = switchbool(StabilizersLeftTryVertical[vertex2]);
					StabilizersLeftTryVertical[vertex3] = switchbool(StabilizersLeftTryVertical[vertex3]);
				}
				if (NewErrorPredictionHorizontal[j])
				{
					//flip stabilizers
					int vertex1 = WestVertex(j, L);
					int vertex2 = EastVertex(j, L);
					int vertex3 = SouthVertex(j, L, H);

					StabilizersLeftTryHorizontal[j] = switchbool(StabilizersLeftTryHorizontal[j]);
					StabilizersLeftTryHorizontal[vertex1] = switchbool(StabilizersLeftTryHorizontal[vertex1]);
					StabilizersLeftTryHorizontal[vertex2] = switchbool(StabilizersLeftTryHorizontal[vertex2]);
					StabilizersLeftTryHorizontal[vertex3] = switchbool(StabilizersLeftTryHorizontal[vertex3]);
				}
				if (NewErrorPredictionBoth[j])
				{
					//flip stabilizers
					int vertex1 = WestVertex(j, L);
					int vertex2 = EastVertex(j, L);
					int vertex3 = SouthVertex(j, L, H);

					StabilizersLeftTryBoth[j] = switchbool(StabilizersLeftTryBoth[j]);
					StabilizersLeftTryBoth[vertex1] = switchbool(StabilizersLeftTryBoth[vertex1]);
					StabilizersLeftTryBoth[vertex2] = switchbool(StabilizersLeftTryBoth[vertex2]);
					StabilizersLeftTryBoth[vertex3] = switchbool(StabilizersLeftTryBoth[vertex3]);
				}
			}

			// find number of stabilizers we have now
			int NumberOfNewStabilizersVertical = 0;
			int NumberOfNewStabilizersHorizontal = 0;
			int NumberOfNewStabilizersBoth = 0;
			for (int j = 0; j < L*H; ++j)
			{
				if (StabilizersLeftTryVertical[j])
				{
					++NumberOfNewStabilizersVertical;
				}
				if (StabilizersLeftTryHorizontal[j])
				{
					++NumberOfNewStabilizersHorizontal;
				}
				if (StabilizersLeftTryBoth[j])
				{
					++NumberOfNewStabilizersBoth;
				}
			}

			// determine option that amounts to minimum number of stabilizers
			int MinNewStabilizers = min(NumberOfNewStabilizersVertical, NumberOfNewStabilizersHorizontal);
			MinNewStabilizers = min(MinNewStabilizers, NumberOfNewStabilizersBoth);

			// Go through with that option
			if (MinNewStabilizers == NumberOfNewStabilizersBoth)
			{
				StabilizersLeft = StabilizersLeftTryBoth;
				ErrorPrediction = ErrorPredictionBoth;
			}
			else if (MinNewStabilizers == NumberOfNewStabilizersVertical)
			{
				StabilizersLeft = StabilizersLeftTryVertical;
				ErrorPrediction = ErrorPredictionVertical;
			}
			else if (MinNewStabilizers == NumberOfNewStabilizersHorizontal)
			{
				StabilizersLeft = StabilizersLeftTryHorizontal;
				ErrorPrediction = ErrorPredictionHorizontal;
			}
		}
		

		
		// ** Find qubit stats
		// Find Qubits that are not predicted after this round
		vector<int> QubitsLeft = Qubits;
		for (int j = 0; j < L*H; j++)
		{
			if (ErrorPrediction[j])
			{
				QubitsLeft[j] = 0;
			}
		}
		// Find Qubits incorrectly predicted from this round
		vector<int> IncorrectQubits;
		for (int i = 0; i < L*H; i++)
		{
			if (ErrorPrediction[i] && !Qubits[i])
			{
				IncorrectQubits.push_back(i);
			}
		}
		// Find Qubits incorrectly predicted from this round
		vector<int> CorrectQubits;
		for (int i = 0; i < L*H; i++)
		{
			if (ErrorPrediction[i] && Qubits[i])
			{
				CorrectQubits.push_back(i);
			}
		}


		// ** See if we are failing
		// Find Number of new stabilziers
		int NumberOfStabilizers = 0;
		for (int i = 0; i < L*H; ++i)
		{
			if (StabilizersLeft[i])
			{
				++NumberOfStabilizers;
			}
		}

		if (NumberOfStabilizers >= NumberOfOldStabilizers)
		{
			if (RoundNumber >= MinTries)
				GiveUp = 1;
		}
		else if (NumberOfStabilizers == 0)
		{
			GiveUp = 1;
		}

		
		
		RoundNumber++;

		//cout << "Round Number = " << RoundNumber << endl;
		
		// ** Print Asymptote
		// Predicted error
		//string title = "P" + to_string(RoundNumber) + ".asy";
		//AsymptotePrint(StabilizersLeft, ErrorPrediction, EmptySetTup, L, H, title);
		// Still to decode
		//title = "S" + to_string(RoundNumber) + ".asy";
		//AsymptotePrintDetectIncorrect(StabilizersLeft, QubitsLeft, ExtendLxH(IncorrectQubits, L, H), EmptySetTup, L, H, title);
		// Correctly predicted
		//title = "C" + to_string(RoundNumber) + ".asy";
		//AsymptotePrintDetectIncorrect(EmptyVec, ExtendLxH(CorrectQubits, L, H), EmptyVec, EmptySetTup, L, H, title);
		//title = "F" + to_string(RoundNumber) + ".asy";
		//AsymptotePrintFull(StabilizersLeft, QubitsLeft, ExtendLxH(CorrectQubits, L, H), ExtendLxH(IncorrectQubits, L, H), L, H, title);
		
	}

	//cout << "\nRound Number: " << RoundNumber << "\n";
	// Calculate if success
	bool Success;
	if (ErrorPrediction == Qubits)
	{
		Success = 1;
		//cout << "\tSUCCESS by OCS\tRound Number = " << RoundNumber << "\n";
	}
	else
	{
		Success = 0;
		//cout << "\tFAILURE by OCS\tRound Number = " << RoundNumber << "\n";
	}
	//cout << "\n\n";

	BoolInt Results;
	Results.A = Success;
	Results.B = RoundNumber;

	return Results;
}


IntInt ComputeFailureNumber(float p, int L, int H, int MinTries, int SampleSize, vector<int> Symmetry, int SymmetrySize, vector<vector<int> > MatrixEdgeWeights)
{

	int FailureNumber =  0;
	int RoundNumberTotal = 0;


	for (int count = 0; count < SampleSize; count++)
	{
		BoolInt Results = TestCode(p, L, H, MinTries, Symmetry, SymmetrySize, MatrixEdgeWeights);
		bool Success = Results.A;
		int RoundNumber = Results.B;

		if (!Success)
		{
			FailureNumber++;
		}
		RoundNumberTotal += RoundNumber;

	}


	//cout << "\nFailNumber: " << FailNumber << endl;
	IntInt Results;
	Results.A = FailureNumber;
	Results.B = RoundNumberTotal;

	return Results;
}


int main()
{

	unsigned int NewSeed = (unsigned int)time(NULL) + 536 * QSUBTRIAL;
	int L = BASHSYSSIZE; 
	float p = BASHpSTART + BASHpINT * BASHpGAP; 
	int MinTries = BASHMINTRIES;
	int SampleSize = BASHSAMPLESIZE;
	
	//unsigned int NewSeed = (unsigned int)time(NULL);
	//int L = 16; // ***
	//float p = 0.27; // ***
	//int MinTries = 0;
	//int SampleSize = 1;

	srand(NewSeed);

	ofstream DataOutput; // Name variable for text file output
	DataOutput.open("QSUBFILENAME.csv");  // ***

	int H = L / 2;

	DataOutput << "L,p,Number of Fails,Total Number of Attempts,Number of Samples" << endl;

	// Create Symmetry
	vector<int> topRow(L, 0);
	topRow[0] = 1;
	vector<int> Symmetry = CreateSymmetry(L, H, topRow);
	int SymmetrySize = static_cast<int>(Symmetry.size());

	vector<vector<int> > MatrixEdgeWeights = create_edge_weight_lookup(L, p, SymmetrySize);

	
	// Test failure rate of this system
	IntInt Results = ComputeFailureNumber(p, L, H, MinTries, SampleSize, Symmetry, SymmetrySize, MatrixEdgeWeights);
	int FailureNumber = Results.A;
	int RoundNumberTotal = Results.B;

	//Output results
	DataOutput << L << "," << p << "," << FailureNumber << "," << RoundNumberTotal << "," << SampleSize << endl;

	DataOutput.close();
	
	return 0;

}
