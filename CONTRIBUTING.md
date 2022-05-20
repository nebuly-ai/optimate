# Guidelines for Contributing to Nebullvm

Hello Coder, 

we are happy you decided to contribute to the library, and we thank you for all the efforts you are going to put in the contribution. Here we briefly expose the main guidelines for uniforming your code to the coding style we adopted for `nebullvm`.

We hope to see you on the PR soon!

Happy coding,

The nebullvm Team


## How to submit an issue
Did you find a bug? Did you have a cool idea you think it should be implemented in nebullvm? Well, GitHub issues are the way you should use for letting us know it!

We do not have a strict policy on issue generation: just use a meaningful title and specify the problem or your proposal in the issue's first comment. Then, you can use the GitHub labels for letting us know the kind of proposal you are doing, e.g `bug` if you are pointing out to a new bug or `enhancement` if you are proposing an improvement of the library. 

## How to contribute to solve an issue
We are always delighted to welcome more people in the section of the Contributors of `nebullvm`! We are excited to welcome you to the family, but before rushing and write 1000 lines of code, please spend few minutes in reading our recommendations for contributing to the library.
* Please fork the library instead of pulling it and creating a new branch
* Work on your fork and when you think the issue is solved open a Pull Request
* In the Pull Request (PR) specify which issues the PR is solving / closing, e.g. if the PR solves issue #1 the comment must be `Closes #1`.
* The PR title must be meaningful and self-explaining

## Coding style
Before git committing and pushing your code you may use `black` for formatting your code. We highly recommend installing `pre-commit` for re-formatting your code when you commit your changes.

For using the nebullvm defined formatting style run the following commands:
```bash
pip install pre-commit black autoflake
pre-commit install
# the following command is optional, but needed if you have already committed some files to your forked repo.
pre-commit run --all-files
```
then add and commit all the changes! Regarding the naming convention we simply use [PEP 8](https://peps.python.org/pep-0008/) for the code and a slight variation of the [Google convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. For the docstrings we redundantly express the input type both in the function definition and the function docstring.