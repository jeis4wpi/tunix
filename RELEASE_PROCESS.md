# Release Process

⚠️ This doc is intended for maintainers of the Tunix library. Steps below
require push access to base repository. However, all are welcome to use this
process for other projects, or suggest improvements!

## Overview

Our release process consists of two main components:

- Adding a new release to the
  [google-tunix](https://pypi.org/project/google-tunix/) project on the Python
  Package Index (pypi).

We follow [semantic versioning](https://semver.org/) for our releases, and have
a different process when releasing major/minor versions (e.g. 1.0 or 1.2) vs a
"patch" release (1.0.1). Both are covered below.

## Creating a new major or minor release

Use the following steps to create an `X.Y.0` release.

1. Similar to the Jax repositories, we keep a named branch `rX.Y` for each minor
   release. We need to set this up.

   If you have not, please set `upstream` as `google/tunix` by running:

   ```shell
   git remote add upstream https://github.com/google/tunix.git
   ```

   From the master branch, create a new branch with a name matching the first
   two digits of the upcoming release:

   ```shell
   git fetch --all
   git checkout --no-track -b rX.Y upstream/master
   git push -u upstream rX.Y
   ```

   This branch will now be used for all subsequent `X.Y.Z` releases, e.g.,
   `0.2.1` should still use branch `r0.2` instead of creating `r0.2.1`.

1. Before we officially push a new stable release to pypi, it is good practice
   to test out a
   [development release](https://pythonpackaging.info/07-Package-Release.html#Versioning-your-code)
   of the package. Development releases will have version numbers like
   `X.Y.0.dev0`, and critically will never be installed by default by `pip`.

   On the relase branch, check the version in `pyproject.toml`. If the current
   version is not `X.Y.0.dev0`, make a PR to update the our version number fo
   look like `X.Y.0.dev0`. This PR should base off our new release branch
   instead of the master branch. You can use the following commands:

   ```shell
   git fetch --all
   git checkout --no-track -b version-bump-X.Y.0.dev0 upstream/rX.Y
   # Update pyproject.toml with an editor.
   git commit -m "Version bump to X.Y.0.dev0"
   git push -u origin version-bump-X.Y.0.dev0
   ```

   On github, make a PR targeting the new release branch instead of the master
   branch, and ask someone to review.

1. On github, we can now create the `X.Y.0.dev0` release.

   This release should be titled `X.Y.0.dev0`, and create a new tag with the
   same name on publish. You can use the following screenshot as a reference.

   ![Release page screenshot](.github/assets/release_screenshot.png)

   Making a github release will automatically kick off a pypi release, as
   configured by [this file](.github/workflows/pypi_release.yml).

1. Wait a few minutes until the release appears on pypi, then test out the
   release by running `pip install google-tunix==X.Y.0.dev0`.

   Try to test the package thoroughly! It is a good idea to run through a few of
   our guides with the new version. Fix any bugs you find, and repeat steps 2
   and 3 with a new dev number (e.g. `X.Y.0.dev1`) until you are confident in
   the release.

   It is important that we make any fixes to the master branch first, and then
   cherry-pick them to the release branch. Given a commit hash `e32e9ded`, you
   can cherry pick a change as follows.

   ```shell
   git checkout rX.Y
   # Make sure we are exactly up to date with the upstream branch.
   git fetch --all
   git reset --hard upstream/rX.Y
   # Cherry pick as many times as you need.
   git cherry-pick e32e9ded
   git push upstream rX.Y
   ```

1. Before cutting the final release, we should try previewing our documentation
   on [tunix.readthedocs.io](tunix.readthedocs.io). This will help catch bugs
   with our symbol export and docstrings.

   The Tunix
   [CONTRIBUTING](https://github.com/google/tunix/blob/main/CONTRIBUTING.md#documentation)
   contains instructions on building and previewing the site, and you can use
   [this PR](https://github.com/google/tunix/pull/278) as a reference for what
   to change. Ask tsbao@ to review.

   During development of the branch, you may use the google-tunix dev mode.
   Remember to update this to the default mode before we merge the PR.

1. We are now ready to cut the official release! Make a PR similar to step 2,
   but updating the release number to `X.Y.0` (no `.dev0` suffix). Land the PR.

   Confirm that the latest commit on our release branch is green before making
   the actual release! We should not release if there are any test failures.

   Make a release similar to step 3, but updating the tag and title to `X.Y.0`.
   Leave "Set as pre-release" unchecked and check the box that says "Set as the
   latest release".

   Click "Publish release" when ready.

1. Now that our release is done, we should bump the version number on our master
   branch. Let `Ŷ = Y + 1`. Our new master branch version should look like
   `X.Ŷ.0.dev0`. We do this so that any accidental master branch releases will
   be dev releases instead of being picked up by pip.

   ```shell
   git fetch --all
   git checkout --no-track -b version-bump-X.Ŷ.0.dev0 upstream/master
   # Update both pyproject.toml with an editor.
   git commit -m "Version bump to X.Ŷ.0.dev0"
   git push -u origin version-bump-X.Ŷ.0.dev0
   ```

   Create a land a PR with this change to the master branch.

## Creating a new patch release

Use the following steps to create a "patch" `X.Y.Z` release. We do this when we
do not yet want to release everything on our master branch, but still would like
to push certain fixes out to our users.

1. We need to bring in code changes to the release branch. Whenever possible
   these should be changes also on the master branch, that we cherry pick for
   the release. Given a commit hash `e32e9ded`, you can cherry pick a change to
   the release branch as follows.

   ```shell
   git checkout rX.Y
   # Make sure we are exactly up to date with the upstream branch.
   git fetch --all
   git reset --hard upstream/rX.Y
   # Cherry pick as many times as you need.
   git cherry-pick e32e9ded
   git push upstream rX.Y
   ```

1. Before we officially push a new stable release to pypi, it is good practice
   to test out a
   [development release](https://pythonpackaging.info/07-Package-Release.html#Versioning-your-code)
   of the package. Development releases will have version numbers like
   `X.Y.Z.dev0`, and critically will never be installed by default by `pip`.

   On the relase branch, check the version in `src/version.py`. If the current
   version is not `X.Y.Z.dev0`, make a PR to update the our version number fo
   look like `X.Y.Z.dev0`. This PR should base off our new release branch
   instead of the master branch. You can use the following commands:

   ```shell
   git fetch --all
   git checkout --no-track -b version-bump-X.Y.Z.dev0 upstream/rX.Y
   # Update both pyproject.toml with an editor.
   git commit -m "Version bump to X.Y.Z.dev0"
   git push -u origin version-bump-X.Y.Z.dev0
   ```

   On github, make a PR from your fork to the new release branch, and ask
   someone to review.

1. On github, we can now create the `X.Y.Z.dev0` release.

   This release should be titled `X.Y.Z.dev0`, and create a new tag with the
   same name on publish. Refer to the screenshot above for details on the github
   release page setup.

   Making a github release will automatically kick off a pypi release, as
   configured by [this file](.github/workflows/publish-to-pypi.yml).

1. Wait a few minutes until the release appears on pypi, then test out the
   release by running `pip install google-tunix==X.Y.Z.dev0`.

   Try to test the package thoroughly! It is a good idea to run through a few of
   our guides with the new version. Fix any bugs you find, and repeat steps 2
   and 3 with a new dev number (e.g. `X.Y.Z.dev1`) until you are confident in
   the release.

1. Before cutting the final release, we should try previewing our documentation
   on tunix.readthedocs.io. This will help catch bugs with our symbol export and
   docstrings.

   The Tunix
   [CONTRIBUTING](https://github.com/google/tunix/blob/main/CONTRIBUTING.md#documentation)
   contains instructions on building and previewing the site, and you can use
   [this PR](https://github.com/google/tunix/pull/278) as a reference for what
   to change. Ask tsbao@ to review.

   During development of the branch, you may use the google-tunix dev mode.
   Remember to update this to the default mode before we merge the PR.

1. We are now ready to cut the official release! Make a PR similar to step 2,
   but updating the release number to `X.Y.Z` (no `.dev0` suffix). Land the PR.

   Confirm that the latest commit on our release branch is green before making
   the actual release! We should not release if there are any test failures.

   Make a release similar to step 3, but updating the tag and title to `X.Y.Z`.
   Leave "Set as pre-release" unchecked and check the box that says "Set as the
   latest release" if `X.Y` is the latest stable release series.

   Click "Publish release" when ready.
