# Guidelines for Contributing to Nebullvm.

Hello coder 👋

We are very happy that you have decided to contribute to the library and we thank you for your efforts. Below we briefly lay out the main guidelines for conforming your code to the coding style we have adopted for `nebullvm`.

We hope to come across your pull request soon!

Happy coding 💫 The nebullvm Team


## How to submit an issue
Did you spot a bug? Did you come up with a cool idea that you think should be implemented in nebullvm? Well, GitHub issues are the best way to let us know!

We don't have a strict policy on issue generation: just use a meaningful title and specify the problem or your proposal in the first problem comment. Then, you can use GitHub labels to let us know what kind of proposal you are making, for example `bug` if you are reporting a new bug or `enhancement` if you are proposing a library improvement. 

## How to contribute to solve an issue
We are always delighted to welcome other people to the contributor section of `nebullvm`! We are looking forward to welcoming you to the community, but before you rush off and write 1000 lines of code, please take a few minutes to read our tips for contributing to the library.
* Please fork the library instead of pulling it and creating a new branch.
* Work on your fork and, when you think the problem has been solved, open a pull request.
* In the pull request specify which problems the it is solving/closing. For instance, if the pull request solves problem #1, the comment should be `Closes #1`.
* The title of the pull request must be meaningful and self-explanatory.


## Coding style
Before you git commit and push your code, please use `black` to format your code. We strongly recommend that you install `pre-commit` to reformat your code when you commit your changes.

To use the formatting style defined for nebullvm, run the following commands:
```bash
pip install pre-commit black autoflake
pre-commit install
# the following command is optional, but needed if you have already committed some files to your forked repo.
pre-commit run --all-files
```
Then add and commit all changes!

As for the naming convention, we follow [PEP 8](https://peps.python.org/pep-0008/) for code and a slight variation of [Google convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. For docstrings we redundantly express the input type in both the function definition and the function docstring.

---

See you soon in the list of nebullvm contributors ✨
