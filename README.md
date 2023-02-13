# q2-plugin-template
A skeleton for QIIME 2 plugins.

To use this template for a new plugin creation, you'll first need to create a new plugin repo by clicking the `Use this template` button.
Fill out the new repository name and description. Following that initial setup, in your new plugin repo you still
need to adjust the following elements:

- exchange `plugin_name` and `plugin-name` with the name of your plugin
- adjust all the required fields in `setup.py` and `plugin_setup.py`
- adjust the developer name in all the copyright headers and the license file
- add `versioneer` support for version management:
    - install `versioneer` [(link)](https://github.com/python-versioneer/python-versioneer) into any python environment: `pip install versioneer`
    - from the project directory, run `versioneer install` to create/modify all the required files
    - add the `# flake8: noqa` flag on top of the `versioneer.py` file for the file to be ignore by lint checks

Enjoy!

### Development using Dev Containers (Visual Studio Code)
To streamline the plugin development process, you can use support for Docker containers built into
VS Code in a form of __Dev Containers__ (see [here](https://code.visualstudio.com/docs/remote/containers) for more information).
In this approach a Docker container will be created and configured (using the provided specification) and your VS Code environment
will be attached to this container where you can continue developing as usual. You only need to specify which version of QIIME 2 you
want to use (see below) and the respective Docker image will be pulled and used to create the container.

The elements you need to use the QIIME 2 devcontainer are:
- container spec located under [.devcontainer.json](.devcontainer.json) in your project directory
- `requirements.txt` file listing all the project-specific conda dependencies that should be installed in the environment
- `prep-dev-container` section in the [Makefile](Makefile)

The only parts that you should need to modify are the packages listed in the `requirements.txt` file. If you are setting up Dev Containers in an already existing repo you will need to copy the files/section listed above.
Once you have your requirements ready follow those steps to get started:
1. Install VS Code (if needed) and enable launch from command line (see [here](https://code.visualstudio.com/docs/setup/mac#_launching-from-the-command-line) for macOS instructions).
2. In your terminal, navigate to the repo you want to open in the devcontainer and execute:
    ```shell
    export QIIME_VERSION=2022.2 && code .
    ```
    Replace `2022.2` with the version of QIIME 2 that you would like to use.
    While it is possible to open those containers without going through the terminal, it is required in our setup to define the desired QIIME version.
3. When prompted, reopen the folder in a Dev Container by clicking on "Reopen in container".
4. The Docker image will be pulled (happens only once per image) and the corresponding container will be created and configured - this may take a few minutes.
5. Once container configuration is finished you should be good to go: you can start developing as usual. The conda environment in the container is already configured and should have your plugin installed inside.

### Development with git hooks and setting up a dev environment
To make your life a bit easier, you can use `git hooks` to automate certain tasks while working with Git (see [here](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) for more details).
Here, we are making use of the `pre-commit` tool which allows us to configure all the hooks in [one place](.pre-commit-config.yaml).
The pre-commit hook configured here will automatically run linting and formatting tasks before any commit (i.e. on `git commit`). In the configuration file
you will see that the pre-push hook is commented-out - it is just more complicated to configure an environment for running that hook (given all the different
conda dependencies that the plugin/QIIME 2 may have).

To set up your dev environment, including the hooks, simply run `make dev` in an existing conda environment - this will install all the required tooling.
You don't need to do anything else - the hooks will be automatically executed whenever you commit.

_Note:_ If, for whatever reason, you need to skip the hooks, you can append the `--no-verify` flag to the git command - the hooks won't be executed.
