"""
Constants we use for AM pre-training
"""

# pylint: skip-file
from am_v2 import config, util

UNKNOWN_PART = 8
UNKNOWN_PART_GUESSED_CUT_TIME = 2
PAD_INDEX = 0
TRUE_INDEX = 1
FALSE_INDEX = 2
SEP_INDEX = 3
IS_CORRECT_MASK_INDEX = 3
IS_ON_TIME_MASK_INDEX = 3
ADD_TASK_MASK_INDEX = 3
FLOAT_TASK_MASK_INDEX = -1
INT_TASK_MASK_INDEX = -1
START_TIME_MASK_INDEX = util.get_start_time_range()[1] + 1
ELAPSED_TIME_MASK_INDEX = config.ARGS.max_elapsed_time + 2
PART_INDEX = 4
DEFAULT_TIME_LIMIT_IN_MS = 43000

STR_TIME = "time"
STR_IS_SELECT_LONG_OPTION = "is_select_long_option"
STR_IS_CONFUSED = "is_confused"
STR_IS_GUESSED = "is_guessed"
STR_NONE = "None"
