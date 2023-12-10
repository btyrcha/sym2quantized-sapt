#!/usr/bin/env bash
# set the default values for local scripts debugging
CI_MERGE_REQUEST_TARGET_BRANCH_NAME=${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-master}

# names only
# filter out deleted files
# compare current commit with merge target
# grab only python files
# trim "\n" and " "
CHANGED_FILES=$(git diff --diff-filter=d --name-only "origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}" | grep -E "\.py$" | tr "\n" " ")
# TODO: exceptions are more tricky as order shouldn't matter
#EXCEPTIONS="src/pybest/linalg/dense.py"
#EXCEPTIONS=""
#CHANGED_FILES=$(echo ${CHANGED_FILES//$EXCEPTIONS/} | tr -d " ")

if [ ! -z "$CHANGED_FILES" ]; then
  echo "Running pylint on ${CHANGED_FILES}"
  python3 -m pylint --rcfile=.pylintrc --output-format=text $CHANGED_FILES
  exit $?
else
  echo "No python files changed skipping MR lint"
fi

# exit success
exit 0
