#!/bin/sh
# Author: abpwrs
# Created: Wed Jun  6 12:01:23 CDT 2018

# input validation
if [[ ${#} -ne 1 ]]; then
# example usage
  echo "Usage: ${0} name_without_spaces"
  echo "Example: ${0} random_forest"
  exit -1
fi


# Standard script setup
FILENAME="$1.job"
# makes output directories in case the structure isn't set up
# uncomment the first mkdir if this is your first time running the script
# mkdir /Users/${USER}/job_output
mkdir /Users/${USER}/job_output/${1}

cat > $FILENAME << EOF
#!/bin/sh
# Author: $USER
# Created: $(date)

# Argon Arguments
# https://wiki.uiowa.edu/display/hpcdocs/Basic+Job+Submission

# set name to be the same as the file
#$ -N ${1}

# set queue (default to hans')
#$ -q HJ

# set max wall-runtime to be one day
#$ -l h_rt=24:00:00

# set parallel env and number of cores (default to 16)
#$ -pe smp 16

# set standard output of file
#$ -o /Users/${USER}/job_output/${1}/${1}_output

# set standard error output file
#$ -e /Users/${USER}/job_output/${1}/${1}_error

# set my email address
#$ -M alexander-powers@uiowa.edu

# set when I want emails (b: beginning, e: end, a: aborted, s: suspended)
#$ -m beas

# script here:
EOF
chmod 700 $FILENAME
