<!-- # Mirror Descent Optimisation -->
This repository hosts the code submitted as part my Individual Project for the 
MSc in Artificial Intelligence at Imperial College London. All code is written in Python3.
<!--For all the dependencies refer to the relevant paragraph of the README file.-->

# Mirror Descent and Interacting Mirror Descent for almost dimension-free optimisation on non-Euclidean spaces

An investigation of Mirror Descent and Interacting particle Mirror Descent optimisation on different problems.
Many fundamental optimisation problems in machine learning and artificial intelligence have at the core of their frameworks the search of an optimal set parameters that lead to the minimisation or maximisation of a differentiable convex or non-convex objective function over either constrained or unconstrained feasible sets. A plethora of these problems utilise iterative methods that compute the gradient of the objective function at each iteration and take a step towards the steepest ascent or descent. A widely used optimisation method is stochastic gradient descent. Stochastic Gradient Descent is typically well-performing for optimising over Euclidean vector spaces. However, Stochastic Gradient Descent  does not always manage to converge to the optimum due to scaling of the dimensionality of the problem on non-Euclidean spaces, or due to noisy computations of the gradient and frequently proves to be expensive if the optimisation is constrained.  In this project we study an other almost dimension-free iterative optimisation algorithm called Mirror Descent and its variant Interacting particles Mirror Descent on six convex frameworks with convex objective functions. Our aim is to observe the performance and  convergence properties  of these methods and also compare them with the standard Stochastic Gradient Descent method. We show that the mirror descent algorithms can outperform Gradient Descent in cases of optimisation over constrained sets with certain geometry, such as the probability simplex. Additionally, we demonstrate the benefits in terms of convergence for allowing particles to interact in the case of Mirror Descent.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Dependencies

Several Python packages are required to be install in order to run this code on your local machine:

```
numpy
torch
scipy
matplotlib
tqdm

networkx
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
