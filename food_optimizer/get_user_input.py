"""Interactive input helpers for reference values and optimization parameters."""

def clean_age_input(raw_input):
    """
    Cleans and validates raw age input.

    Parameters:
    - raw_input (str | float | int): Raw input representing age.

    Returns:
    - int: Rounded age, capped at 120 if greater.

    Raises:
    - ValueError: If input is not a number ≥ 1.
    """
    try:
        age_float = float(raw_input)
    except ValueError:
        raise ValueError("Invalid age input. Must be a number ≥ 1.")

    if age_float < 1:
        raise ValueError("Age cannot be less than 1.")

    age = round(age_float)
    return min(age, 120)

def clean_sex_input(raw_input):
    """
    Cleans and validates sex input.

    Parameters:
    - raw_input (str): User input for sex.

    Returns:
    - str: 'man' or 'woman'.

    Raises:
    - ValueError: If input is not one of the allowed values.
    """
    val = raw_input.strip().lower()
    if val in ["man", "woman"]:
        return val
    raise ValueError("Sex must be 'man' or 'woman'.")

def clean_pregnant_input(raw_input):
    """
    Cleans and validates pregnancy status input.

    Parameters:
    - raw_input (str): User input ("yes" or "no").

    Returns:
    - bool: True if pregnant, False otherwise.

    Raises:
    - ValueError: If input is not 'yes' or 'no'.
    """
    val = raw_input.strip().lower()
    if val == "yes":
        return True
    elif val == "no":
        return False
    raise ValueError("Pregnant input must be 'yes' or 'no'.")

def clean_trimester_input(raw_input):
    """
    Cleans and validates trimester input.

    Parameters:
    - raw_input (str | int): User input indicating trimester.

    Returns:
    - int: 1, 2, or 3

    Raises:
    - ValueError: If input is not 1, 2, or 3.
    """
    try:
        val = int(raw_input)
        if val in [1, 2, 3]:
            return val
        raise ValueError
    except:
        raise ValueError("Trimester must be 1, 2, or 3.")

def clean_breastfeeding_input(raw_input):
    """
    Cleans and validates breastfeeding status input.

    Parameters:
    - raw_input (str): User input ("yes" or "no").

    Returns:
    - bool: True if breastfeeding, False otherwise.

    Raises:
    - ValueError: If input is not 'yes' or 'no'.
    """
    val = raw_input.strip().lower()
    if val == "yes":
        return True
    elif val == "no":
        return False
    raise ValueError("Breastfeeding input must be 'yes' or 'no'.")

def clean_breastfeeding_phase_input(raw_input):
    """
    Converts breastfeeding phase input from 'yes'/'no' to phase 1 or 2.

    Parameters:
    - raw_input (str): User input ("yes" for early phase, "no" for later).

    Returns:
    - int: 1 for early phase, 2 for later phase.

    Raises:
    - ValueError: If input is not 'yes' or 'no'.
    """
    val = raw_input.strip().lower()
    if val == "yes":
        return 1
    elif val == "no":
        return 2
    raise ValueError("Breastfeeding phase input must be 'yes' or 'no'.")

def clean_num_solutions_input(raw_input):
    """Validate and convert num_solutions to int ≥ 1."""
    try:
        val = int(raw_input)
        if val >= 1:
            return val
        raise ValueError
    except:
        raise ValueError("Number of solutions must be an integer ≥ 1.")

def clean_prioritize_input(raw_input):
    """Validate prioritize input to be 'weight', 'toxins', or 'none'."""
    val = raw_input.strip().lower()
    if val in ['weight', 'toxins', 'none']:
        return None if val == 'none' else val
    raise ValueError("Prioritize must be 'weight', 'toxins', or 'none'.")

def clean_max_foods_input(raw_input):
    """Validate and convert max_foods to int ≥ 1."""
    try:
        val = int(raw_input)
        if val >= 1:
            return val
        raise ValueError
    except:
        raise ValueError("Maximum number of foods must be an integer ≥ 1.")

def clean_prefer_sparse_input(raw_input):
    """Validate yes/no to boolean for prefer_sparse."""
    val = raw_input.strip().lower()
    if val == 'yes':
        return True
    elif val == 'no':
        return False
    raise ValueError("Prefer sparse input must be 'yes' or 'no'.")


def age_inquiry():
    """
    Prompts the user to input their age and returns the validated value.

    Returns:
    - int: Validated and rounded age input (capped at 120).
    """
    raw = input("State your age: ")
    return clean_age_input(raw)

def sex_inquiry():
    """
    Prompts the user to specify their sex.

    Returns:
    - str: 'man' or 'woman' based on validated input.
    """
    raw = input("Are you man or woman? ")
    return clean_sex_input(raw)

def pregnant_inquiry():
    """
    Asks the user if they are pregnant.

    Returns:
    - bool: True if pregnant, False otherwise.
    """
    raw = input("Are you pregnant? (yes/no): ")
    return clean_pregnant_input(raw)

def trimester_inquiry():
    """
    Prompts the user to specify their pregnancy trimester.

    Returns:
    - int: Trimester number (1, 2, or 3).
    """
    raw = input("State which trimester you are in (1, 2, 3): ")
    return clean_trimester_input(raw)

def breastfeeding_inquiry():
    """
    Asks the user if they are currently breastfeeding.

    Returns:
    - bool: True if breastfeeding, False otherwise.
    """
    raw = input("Are you breastfeeding? (yes/no): ")
    return clean_breastfeeding_input(raw)

def breastfeeding_phase_inquiry():
    """
    Asks if the baby is younger than six months to determine breastfeeding phase.

    Returns:
    - int: 1 if baby is younger than six months, 2 otherwise.
    """
    raw = input("Is the baby younger than six months? (yes/no): ")
    return clean_breastfeeding_phase_input(raw)


def num_solutions_inquiry():
    raw = input("How many solutions would you like to generate? (integer ≥ 1): ")
    return clean_num_solutions_input(raw)

def prioritize_inquiry():
    raw = input("Do you want to prioritize 'weight', 'toxins', or 'none'?: ")
    return clean_prioritize_input(raw)

def max_foods_inquiry():
    raw = input("What is the maximum number of foods per solution? (integer ≥ 1): ")
    return clean_max_foods_input(raw)

def prefer_sparse_inquiry():
    raw = input("Do you prefer sparse solutions (fewer foods)? (yes/no): ")
    return clean_prefer_sparse_input(raw)

def get_ref_val_input_parameters():
    """
    Gathers user input interactively for all parameters needed to build a nutritional profile.

    Returns:
    - dict: Dictionary with the following keys and user-provided values:
        - 'sex' (str): 'man' or 'woman'
        - 'pregnant' (bool or None): Whether the user is pregnant
        - 'trimester' (int or None): Trimester if pregnant (1, 2, or 3)
        - 'breastfeeding' (bool or None): Whether the user is breastfeeding
        - 'phase' (int or None): Breastfeeding phase (1 if <6 months, 2 otherwise)
        - 'age' (int): Age in years (rounded and capped at 120)
    """
    dict_input_parameters = {key: None for key in [
        'sex', 'pregnant', 'trimester', 'breastfeeding', 'phase', 'age']}
    
    # Sex parameter
    sex = sex_inquiry()
    dict_input_parameters['sex'] = sex

    # Additional parameters if user is a woman
    if sex == 'woman':
        pregnant = pregnant_inquiry()
        dict_input_parameters['pregnant'] = pregnant
        if pregnant:
            trimester = trimester_inquiry()
            dict_input_parameters['trimester'] = trimester
        else:
            breastfeeding = breastfeeding_inquiry()
            dict_input_parameters['breastfeeding'] = breastfeeding
            if breastfeeding:
                phase = breastfeeding_phase_inquiry()
                dict_input_parameters['phase'] = phase

    # Age parameter
    age = age_inquiry()
    dict_input_parameters['age'] = age

    return dict_input_parameters


def get_opt_input_parameters():

    dict_input_parameters = {key: None for key in [
        'num_solutions', 'prioritize', 'max_foods', 'prefer_sparse'
    ]}

    # Optimization-related inputs
    dict_input_parameters['num_solutions'] = num_solutions_inquiry()
    dict_input_parameters['prioritize'] = prioritize_inquiry()
    dict_input_parameters['max_foods'] = max_foods_inquiry()
    dict_input_parameters['prefer_sparse'] = prefer_sparse_inquiry()

    return dict_input_parameters


# Only run interactively if script is called directly
if __name__ == "__main__":
    get_ref_val_input_parameters()
    get_opt_input_parameters()
