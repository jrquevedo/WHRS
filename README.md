# WHRS
This software implements a simulation and a optimization of a combined waste heat recovery system (WHRS) composed of four subsystems as is described in the article:</p>

_Díaz-Secades, L. A., González, R., Rivera, N., Montañés, E., & Quevedo, J. R. (2023). Waste heat recovery system for marine engines optimized through a preference learning rank function embedded into a Bayesian optimizer. Ocean Engineering, 281, 114747._

There are two implementations of the WHRS simulator used for different goals.

## WHRS_Matlab
This software is distributed as a MATLAB R2021b project in an executable file: WHRS_System.mlx. The following sections describe the process.

Input data

When the Matlab is run, a first dialogue box will appear. In here, particulars of the engine should be written. When finished, press OK button and a second dialogue box will appear. In this second box, characteristics of the waste heat recovery system should be written. When finished, press OK button and the program will make calculations.

Output data

Once all the variables are load into the program, the code will calculate performance of the system. Output data will appear on the screen as a dialogue box and will be saved in an Excel document named WHRS_results.xlsx as well.

Warning messages

The program has several warnings that avoid the system to operate out of range, programmed warnings are:

-          Exhaust gas temperature

-          Exhaust gas mass flow

-          Jacket water temperature

-          Jacket water pressure

-          Sea water temperature

-          Sea water pressure

-          Thermoelectric hot side temperature > 200 ºC

-          Thermoelectric cold side temperature bigger than hot side temperature

-          Evaporator chamber pressure

-          Evaporator pinch point

-          Rankine cycle superheat and subcool

-          Organic Rankine cycle superheat and subcool


## WHRS_Python
This software is distributed as a [Spyder](https://www.spyder-ide.org/) Proyect.
There are two executable files: `learnRanking.py` and `optimizeModel.py`.
The next sections describe the process.

### Input data
Several random WHRS inputs where generated and the Load (L), Exergy efficiency (E), and the Electricity Production Cost(C) where generated.

The doubtless pairs are stored in `PreOrderedPairs*.csv`. The first pair is prefered to the second.
A selection of the doubt pairs were shown to the experts. Their preferences were stored in `UserOrderedPairs*.csv`. The preference is marked as `A` (first pair is better), `B` (second pair is better) or `X` (no decision).

### Learn a ranking
To execute this stage execute `learnRanking.py`.

From the `UserOrderedPairs*.csv` and the same number pairs from `PreOrderedPairs*.csv` a data set id generated.
A linear SVM is used to implement a ranking learing procedure that generates a linear model that will be stored in `rankModel*.csv`.
A cross validation is performed to estimate the model's error.

### Optimize the WHRS
To execute this stage execute `optimizeModel.py`.

A Bayesian optimization procedure is used in this stage to carry out the optimizations.

For all optimization batchs the best WHRS state for a load interval is obtained. The load interval vary from 60% to 100%.

The first optimization batch consists on optimize each output variable (L,E,C) independently. The results show that the WHRS's state that maximizes a varaible does not maximizes the other two.

The second optimization batch optimizes the rank model that combines L, E and C in the way that experts indicate with their preferences.

The third optimization batch optimizes the rank model when vary the C's influence in the experts' decision.
