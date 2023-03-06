# Guidelines for Contributing to Nebullvm 🚀

Hello coder 👋

We are very happy that you have decided to contribute to the library and we thank you for your efforts. Here you can find guidelines on how to standardize your code with the style we adopted for `nebullvm`.  But remember, there are various ways to help the community other than submitting code contributions, answering questions and improving the documentation are also very valuable.

It also helps us if you mention our library in your blog posts to show off the cool things it's made possible, or just give the repository a ⭐️ to show us that you appreciate the project

This guide was inspired by the awesome [Transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) guide to contributing.

We hope to come across your pull request soon!

Happy coding 💫 The nebullvm Team


## How to submit an issue
Did you spot a bug? Did you come up with a cool idea that you think should be implemented in nebullvm? Well, GitHub issues are the best way to let us know!

We don't have a strict policy on issue generation, just use a meaningful title and specify the problem or your proposal in the first problem comment. Then, you can use GitHub labels to let us know what kind of proposal you are making, for example `bug` if you are reporting a bug or `enhancement` if you are proposing a library improvement. 

## How to contribute to solve an issue
We are always delighted to welcome other people to the contributors section of nebullvm! We are looking forward to welcoming you to the community, here are some guidelines to follow:
1. Please [fork](https://github.com/nebuly-ai/nebullvm/fork) the [library](https://github.com/nebuly-ai/nebullvm) by clicking on the Fork button on the repository's page. This will create a copy of the repository in your GitHub account.
2. Clone your fork to your local machine, and add the base repository as a remote:
    ```bash
    $ git clone git@github.com:<your Github handle>/nebuly-ai/nebullvm.git
    $ cd nebullvm
    $ git remote add upstream https://github.com/nebuly-ai/nebullvm.git
    ```
3. Install the library in editable mode with the following command:
    ```bash
    $ pip install -e .
    ```
4. Work on your fork to develop the feature you have in mind.
5. Nebullvm relies on `black` to format its source code consistently. To use the formatting style defined for nebullvm, run the following commands:
    ```bash
    $ pip install pre-commit black autoflake
    $ pre-commit install
    # the following command is optional, but needed if you have already 
    # committed some files to your forked repo.
    $ pre-commit run --all-files
    ```
    As for the naming convention, we follow [PEP 8](https://peps.python.org/pep-0008/) for code and a slight variation of [Google convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings. For docstrings we redundantly express the input type in both the function definition and the function docstring.
6. Once you're happy with your changes, add changed files with git add and commit your code:
    ```bash
    $ git add edited_file.py
    $ git commit -m "Add a cool feature"
    ```
7. Push your changes to your repo:
    ```bash
    $ git push
    ```
8. Now you can go to the repo you have forked on your github profile and press on **Pull Request** to open a pull request. In the pull request specify which problems it is solving. For instance, if the pull request solves `Issue #1`, the comment should be `Closes #1`. Also make the title of the pull request meaningful and self-explanatory.
---

See you soon in the list of nebullvm contributors 🌈
