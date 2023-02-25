# WHRS
This software implements a simulation of a combined waste heat recovery system (WHRS) composed of four subsystems as is described in the article:</p>

_Díaz-Secades, Luis Alfonso; González, R; Rivera, N; Montañés, Elena; Quevedo, José Ramón. A novel waste heat recovery system for a medium speed marine engine optimized through a preference learning rank function embedded into a bayesian optimizer. Ocean Engineering. (Under Review)_

There are two implementations of the WHRS simulator used for different goals.

## WHRS_Matlab

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
A Bayesian optimization procedure is used in this stage to carry out the optimizations.

For all optimization batchs the best WHRS state for a load interval is obtained. The load interval vary from 60% to 100%.

The first optimization batch consists on optimize each output variable (L,E,C) independently. The results show that the WHRS's state that maximizes a varaible does not maximizes the other two.

The second optimization batch optimizes the rank model that combines L, E and C in the way that experts indicate with their preferences.

The third optimization batch optimizes the rank model when vary the C's influence in the experts' decision.
