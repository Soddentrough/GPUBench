#!/bin/sh
# Wrapper to invoke rpmbuild in a sanitized environment.
# Unsets broken environment module function exports (e.g. from Lmod on Fedora/RHEL)
# that cause /bin/sh -e inside rpmbuild scripts to abort with bad file descriptor/syntax errors.
exec env -u BASH_FUNC_module%% -u BASH_FUNC_ml%% -u BASH_FUNC__module_raw%% rpmbuild "$@"
