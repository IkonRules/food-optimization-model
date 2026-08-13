"""Map Swedish Market Basket Study toxin categories onto individual food items."""

import re

import pandas as pd

from food_optimizer.config import DATA_DIR


input_path_foods = DATA_DIR / "df_food_database.csv"
input_path_toxins = DATA_DIR / "df_matkorgen_toxins_database.csv"

food_database_swe = pd.read_csv(input_path_foods)
df_toxins = pd.read_csv(input_path_toxins)

# Dictionary containing info about categorization imits for easy update of parameters.
category_limits_dict = {'Fat-/Lean fish limit': 8,
                        'Fat-/Lean dairy limit': 13} 

def print_categories():
    """
    Prints the current food and toxin database categories to the console.
    Assumes `categories_food_database_swe` and `categories_df_toxins` are defined in scope.
    """
    print('Food database categories:\n\n', categories_food_database_swe, 
          '\n\nMatkorgen database categories:\n\n', categories_df_toxins)

def move_column(df, col_name, new_index):
    """
    Moves a column to a specified position in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - col_name (str): The column name to move.
    - new_index (int): The new index position for the column.

    Returns:
    - pd.DataFrame: Updated DataFrame with reordered columns.
    """
    cols = df.columns.tolist()
    cols.insert(new_index, cols.pop(cols.index(col_name)))
    return df[cols]

def cell_contains(cell, words, match_type="any", match_whole_words=True):
    """
    Checks if a string contains one or more words.

    Parameters:
    - cell (str): Text to check.
    - words (list): List of words to match.
    - match_type (str): 'any' for partial match, 'all' for full match (default: 'any').
    - match_whole_words (bool): Whether to match whole words only (default: True).

    Returns:
    - bool: True if match condition is met.
    """
    if not isinstance(cell, str):
        return False

    cell = cell.lower()

    if match_whole_words:
        matches = [re.search(rf"\b{re.escape(word)}\b", cell) for word in words]
    else:
        matches = [re.search(re.escape(word), cell) for word in words]

    if match_type == "any":
        return any(matches)
    elif match_type == "all":
        return all(matches)
    else:
        raise ValueError("match_type must be 'any' or 'all'")

def df_category_items(df, category):
    """
    Displays all food items in a DataFrame that belong to a given category.

    Parameters:
    - df (pd.DataFrame): DataFrame with a 'Gruppering' or 'Product type' column.
    - category (str): Category name to filter by.

    Returns:
    - pd.DataFrame: Filtered view of items in the selected category.
    """
    with pd.option_context('display.max_rows', None):
        if 'Gruppering' in df.columns:
            return df[df['Gruppering'] == category][['Livsmedelsnamn', 'Gruppering', 'Category']]
        elif 'Product type' in df.columns:
            return df[df['Product type'] == category]
        else:
            print("Unknown DataFrame format")
            return

def assign_single_value(food_name, category):
    """
    Assigns a single category value to a specific food item in the food database.

    Parameters:
    - food_name (str): Name of the food item.
    - category (str): Category to assign.
    """
    food_database_swe.loc[food_database_swe['Livsmedelsnamn'] == food_name, 'Category'] = category

def assign_multiple_single_values(list_of_food_names, category):
    """
    Assigns the same category to multiple food items.

    Parameters:
    - list_of_food_names (list): List of food item names.
    - category (str): Category to assign.
    """
    for food_name in list_of_food_names:
        assign_single_value(food_name, category)

def annotate_unassigned_processed(processed_categories):
    """
    Tags food items in processed groups with a temporary 'Processed but unassigned' label
    if they currently lack a category.

    Parameters:
    - processed_categories (list): List of grupperings considered processed.
    """
    food_database_swe['Category'] = food_database_swe.apply(
        lambda row: (
            'Processed but unassigned' 
            if row['Gruppering'] in processed_categories and pd.isna(row['Category'])
            else row.get('Category', row['Gruppering'])
        ), axis=1)

def find_NaNs_in_gruppering(gruppering):
    """
    Finds all NaN category values within a specific 'Gruppering'.

    Parameters:
    - gruppering (str): Gruppering to search in.

    Returns:
    - pd.DataFrame: Filtered rows with NaN in 'Category'.
    """
    return food_database_swe[
        (food_database_swe['Gruppering'] == f'{gruppering}') & 
        (food_database_swe['Category'].isna())
    ]

def search_df_for_words(words, specific_gruppering=False, df=food_database_swe, match_type='any', match_whole_words=True):
    """
    Searches for rows where 'Livsmedelsnamn' contains any or all specified words.

    Parameters:
    - words (list): Words to match.
    - specific_gruppering (str or False): Optional filter by 'Gruppering'.
    - df (pd.DataFrame): DataFrame to search.
    - match_type (str): 'any' or 'all' (default: 'any').
    - match_whole_words (bool): Match full words only (default: True).

    Returns:
    - pd.DataFrame: Matching rows.
    """
    matches = df['Livsmedelsnamn'].apply(
        lambda x: cell_contains(x, words, match_type, match_whole_words))
    matching_rows = df[matches]

    if not specific_gruppering:
        return matching_rows
    else:
        return matching_rows[matching_rows['Gruppering'] == f'{specific_gruppering}']

def print_remaining_categories():
    """
    Prints categories in the food database that remain to be processed.
    Assumes `categories_food_database_swe_remaining` and `categories_df_toxins` are defined in scope.
    """
    print('Food database categories remaining:\n\n', categories_food_database_swe_remaining, 
          '\n\nMatkorgen database categories:\n\n', categories_df_toxins)

def map_products_to_category(
    df: pd.DataFrame,
    grouping: str | list = None,
    conditions: list = None,
    include_words: list = None,
    exclude_words: list = None,
    match_whole_words: bool = True,
    word_match_type: str = 'any'
) -> pd.Series:
    """
    Returns a boolean mask for rows in `df` that match grouping and additional word conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to operate on.
    - grouping (str or list, optional): One or more 'Gruppering' values to filter.
    - conditions (list of tuples, optional): Extra conditions like [('Fett, totalt', '>', 10)].
    - include_words (list, optional): List of words that should appear in 'Livsmedelsnamn'.
    - exclude_words (list, optional): List of words that should NOT appear in 'Livsmedelsnamn'.
    - match_whole_words (bool): If True, matches whole words; else substrings.
    - word_match_type (str): 'any' or 'all' for matching logic.

    Returns:
    - pd.Series (bool): Boolean mask to be used for assigning categories.
    """

    mask = pd.Series(True, index=df.index)

    if grouping is not None:
        if isinstance(grouping, list):
            mask &= df['Gruppering'].isin(grouping)
        else:
            mask &= df['Gruppering'] == grouping

    if conditions:
        for col, op, val in conditions:
            if op == '>':
                mask &= df[col] > val
            elif op == '>=':
                mask &= df[col] >= val
            elif op == '<':
                mask &= df[col] < val
            elif op == '<=':
                mask &= df[col] <= val
            elif op == '==':
                mask &= df[col] == val
            elif op == '!=':
                mask &= df[col] != val

    def _contains_words(cell, words, match_type, whole_words):
        if pd.isna(cell):
            return False
        cell = str(cell).lower()
        if whole_words:
            wordlist = cell.split()
        else:
            wordlist = [cell]
        matches = [any(word in token for token in wordlist) for word in words]
        return all(matches) if match_type == 'all' else any(matches)

    if include_words:
        mask &= df['Livsmedelsnamn'].apply(
            lambda x: _contains_words(x, include_words, word_match_type, match_whole_words)
        )

    if exclude_words:
        mask &= ~df['Livsmedelsnamn'].apply(
            lambda x: _contains_words(x, exclude_words, word_match_type, match_whole_words)
        )

    return mask


# Remove unnecessary products.
values_to_remove = ['Rätter', 'Smaksättare']
food_database_swe = food_database_swe[~food_database_swe['Gruppering'].isin(values_to_remove)]
food_database_swe = food_database_swe.dropna(subset=['Gruppering'])

# Add "Category" column to food database.
food_database_swe['Category'] = None

# Move "Category for better structure"
food_database_swe = move_column(food_database_swe, 'Category', 2)

# Create lists of with the sets of categories for overview.
categories_food_database_swe = food_database_swe['Gruppering'].unique().tolist()
categories_df_toxins = df_toxins.columns.tolist()
categories_df_toxins = [x for x in categories_df_toxins if x not in categories_df_toxins[:1]] 

# print_categories()

mask_fats_and_oils = map_products_to_category(
    df=food_database_swe,
    grouping='Fett, olja'
)

# Apply category
food_database_swe.loc[mask_fats_and_oils, 'Category'] = 'Fats and oils'


limit = category_limits_dict['Fat-/Lean dairy limit']

# Fatty dairy: Mejeri with fat > limit
mask_fatty_mejeri = map_products_to_category(
    df=food_database_swe,
    grouping='Mejeri',
    conditions=[('Fett, totalt', '>', limit)]
)

# Fatty dairy: Pålägg with fat > limit and name contains ost/smör/mjölk
mask_fatty_pålägg = map_products_to_category(
    df=food_database_swe,
    grouping='Pålägg',
    conditions=[('Fett, totalt', '>', limit)],
    include_words=['ost', 'smör', 'mjölk'],
    match_whole_words=False,
    word_match_type='any'
)

# Lean dairy: Mejeri with fat ≤ limit
mask_lean_mejeri = map_products_to_category(
    df=food_database_swe,
    grouping='Mejeri',
    conditions=[('Fett, totalt', '<=', limit)]
)

# Lean dairy: Pålägg with fat ≤ limit and name contains ost/smör/mjölk
mask_lean_pålägg = map_products_to_category(
    df=food_database_swe,
    grouping='Pålägg',
    conditions=[('Fett, totalt', '<=', limit)],
    include_words=['ost', 'smör', 'mjölk'],
    match_whole_words=False,
    word_match_type='any'
)

# Apply category assignments
food_database_swe.loc[mask_fatty_mejeri | mask_fatty_pålägg, 'Category'] = 'Fatty dairy products'
food_database_swe.loc[mask_lean_mejeri | mask_lean_pålägg, 'Category'] = 'Lean dairy products'


limit = category_limits_dict['Fat-/Lean fish limit']

# Fatty fish: Fisk, skaldjur with fat > limit
mask_fatty_fish = map_products_to_category(
    df=food_database_swe,
    grouping='Fisk, skaldjur',
    conditions=[('Fett, totalt', '>', limit)]
)

# Fatty fish: Ägg, rom, kaviar with fat > limit and contains 'rom' or 'kaviar'
mask_fatty_egg = map_products_to_category(
    df=food_database_swe,
    grouping='Ägg, rom, kaviar',
    conditions=[('Fett, totalt', '>', limit)],
    include_words=['rom', 'kaviar'],
    match_whole_words=False,
    word_match_type='any'
)

# Lean fish: Fisk, skaldjur with fat ≤ limit
mask_lean_fish = map_products_to_category(
    df=food_database_swe,
    grouping='Fisk, skaldjur',
    conditions=[('Fett, totalt', '<=', limit)]
)

# Lean fish: Ägg, rom, kaviar with fat ≤ limit and contains 'rom' or 'kaviar'
mask_lean_egg = map_products_to_category(
    df=food_database_swe,
    grouping='Ägg, rom, kaviar',
    conditions=[('Fett, totalt', '<=', limit)],
    include_words=['rom', 'kaviar'],
    match_whole_words=False,
    word_match_type='any'
)

# Apply category assignments
food_database_swe.loc[mask_fatty_fish | mask_fatty_egg, 'Category'] = 'Fatty fish'
food_database_swe.loc[mask_lean_fish | mask_lean_egg, 'Category'] = 'Lean fish'

mask_eggs = map_products_to_category(
    df=food_database_swe,
    grouping='Ägg, rom, kaviar',
    include_words=['ägg', 'äggula', 'äggvita', 'äggröra'],
    word_match_type='any',
    match_whole_words=True
)

# Apply category assignment
food_database_swe.loc[mask_eggs, 'Category'] = 'Eggs'

mask_meat = map_products_to_category(
    df=food_database_swe,
    grouping=['Kött', 'Kyckling, fågel', 'Lever, njure, tunga etc.']
)

# Apply category assignment
food_database_swe.loc[mask_meat, 'Category'] = 'Meat'

vegan_indication_words = ['vegansk', 'vegetarisk', 'veg', 'veg.']

# Block 1: Category = 'Meat', contains salted words, excludes vegan words
mask_1 = (
    (food_database_swe['Category'] == 'Meat') &
    map_products_to_category(
        df=food_database_swe,
        grouping=None,
        include_words=['m. salt', 'salt', 'extra salt', 'saltad'],
        exclude_words=vegan_indication_words,
        match_whole_words=True
    )
)

# Block 2: Gruppering not in list, contains meat terms, excludes vegan words
mask_2 = (
    ~food_database_swe['Gruppering'].isin(['Rätter', 'Quorn, sojaprotein, vegetariska produkter']) &
    map_products_to_category(
        df=food_database_swe,
        grouping=None,
        include_words=['bacon', 'korv', 'salami', 'leverpastej', 'skinka'],
        exclude_words=vegan_indication_words,
        match_whole_words=True
    )
)

# Block 3: Gruppering not in extended list, contains 'rökt', excludes vegan words
mask_3 = (
    ~food_database_swe['Gruppering'].isin([
        'Rätter', 'Quorn, sojaprotein, vegetariska produkter',
        'Fisk, skaldjur', 'Ägg, rom, kaviar'
    ]) &
    map_products_to_category(
        df=food_database_swe,
        grouping=None,
        include_words=['rökt'],
        exclude_words=vegan_indication_words,
        match_whole_words=True
    )
)

# Block 4: Gruppering == 'Pålägg', name contains meat keywords (partial match), excludes vegan words
mask_4 = map_products_to_category(
    df=food_database_swe,
    grouping='Pålägg',
    include_words=['korv', 'kalkon', 'kött', 'skinka'],
    exclude_words=vegan_indication_words,
    match_whole_words=False
)

# Combine all masks
mask_processed_meat = mask_1 | mask_2 | mask_3 | mask_4

# Assign category
food_database_swe.loc[mask_processed_meat, 'Category'] = 'Processed meat'

# Block 1: Direct match on Gruppering
mask_quorn_group = map_products_to_category(
    df=food_database_swe,
    grouping='Quorn, sojaprotein, vegetariska produkter'
)

# Block 2: Glass products with 'soja' or 'havre' in name (substring match)
mask_glass_soja_havre = map_products_to_category(
    df=food_database_swe,
    grouping='Glass',
    include_words=['soja', 'havre'],
    match_whole_words=False,
    word_match_type='any'
)

# Combine both masks
mask_meat_substitutes = mask_quorn_group | mask_glass_soja_havre

# Apply assignment
food_database_swe.loc[mask_meat_substitutes, 'Category'] = 'Meat substitutes'

common_fruits_list = [
    'Apelsin','Äpple', 'äppel', 'Citron','Lime','Jordgubb', 'jordgubbs', 'Banan','Mango','Ananas','Päron',
    'Hallon','Blåbär', 'blåbärs','Svarta vinbär', 'svarta vinbärs','Tranbär', 'tranbärs','Passionsfrukt',
    'passionsfrukts', 'Vattenmelon', 'Granatäpple', 'Kiwi', 'Druva', 'druv', 'vindruvs', 'Persika', 'persiko', 
    'Grapefrukt', 'frukt'
]

# Block 1: Whole-word matches in beverage names
mask_bev_1 = map_products_to_category(
    df=food_database_swe,
    grouping='Dryck',
    include_words=[
        'öl', 'läsk', 'mineralvatten', 'mineral vatten', 'cider', 
        'kolsyrad', 'kolsyrat', 'frukt', 'bär', 'juice', 'smoothie'
    ],
    match_whole_words=True,
    word_match_type='any'
)

# Block 2: Substring matches for plant-based beverages
mask_bev_2 = map_products_to_category(
    df=food_database_swe,
    grouping='Dryck',
    include_words=['mandel', 'havre', 'soja'],
    match_whole_words=False,
    word_match_type='any'
)

# Block 3: Must contain both 'juice' AND a common fruit name (substrings)
mask_juice = map_products_to_category(
    df=food_database_swe,
    grouping='Dryck',
    include_words=['juice'],
    match_whole_words=False,
    word_match_type='any'
)
mask_fruit = map_products_to_category(
    df=food_database_swe,
    grouping='Dryck',
    include_words=common_fruits_list,
    match_whole_words=False,
    word_match_type='any'
)
mask_bev_3 = mask_juice & mask_fruit

# Combine all beverage masks
mask_beverages = mask_bev_1 | mask_bev_2 | mask_bev_3

# Apply category
food_database_swe.loc[mask_beverages, 'Category'] = 'Beverages'

# Block 1: Grupp is in list of cereal product groups
mask_group_cereals = map_products_to_category(
    df=food_database_swe,
    grouping=[
        'Flingor, frukostflingor, müsli, gröt, välling',
        'Pasta, ris, gryn'
    ]
)

# Block 2: Grupp is Bröd and name contains 'bröd' (substring)
mask_bröd = map_products_to_category(
    df=food_database_swe,
    grouping='Bröd',
    include_words=['bröd'],
    match_whole_words=False,
    word_match_type='any'
)

# Block 3: Grupp is Mjöl and name contains 'mjöl' (substring)
mask_mjöl = map_products_to_category(
    df=food_database_swe,
    grouping='Mjöl',
    include_words=['mjöl'],
    match_whole_words=False,
    word_match_type='any'
)

# Combine all blocks
mask_cereal_products = mask_group_cereals | mask_bröd | mask_mjöl

# Apply category
food_database_swe.loc[mask_cereal_products, 'Category'] = 'Cereal products'

# Manual assignment for specific food items
assign_multiple_single_values(
    ['Tacoskal', 'Skorpor fullkorn osötade', 'Skorpor vete osötade'],
    'Cereal products'
)

# Block 1: Gruppering is Potatis
mask_potatis_group = map_products_to_category(
    df=food_database_swe,
    grouping='Potatis'
)

# Block 2: Name contains both 'chips' and 'potatis' (whole words)
mask_potatis_words = map_products_to_category(
    df=food_database_swe,
    include_words=['chips', 'potatis'],
    match_whole_words=True,
    word_match_type='all'
)

# Combine masks
mask_potatoes = mask_potatis_group | mask_potatis_words

# Apply category
food_database_swe.loc[mask_potatoes, 'Category'] = 'Potatoes'

mushrooms_to_exclude = [
    'kantarell', 'kantareller', 'trattkantarell', 'trattkantareller', 'karljohan', 'karljohansvampar', 
    'björksopp', 'björksoppar', 'smörsopp', 'smörsoppar', 'rödgul trumpetsvamp', 'rödgula trumpetsvampar',
    'taggsvamp', 'taggsvampar', 'riskasvamp', 'riskor', 'champinjon', 'champinjoner', 'kremla', 'kremlor'
]

baljväxter_to_exclude = [
    'kikärta', 'kikärtor', 'sojaböna', 'sojabönor', 'bondböna', 'bondbönor', 'svart böna', 'svarta bönor', 
    'kidneyböna', 'kidneybönor', 'vit böna', 'vita bönor', 'mungböna', 'mungbönor'
]

# Define mask
mask_vegetables = map_products_to_category(
    df=food_database_swe,
    grouping='Grönsaker, baljväxter, svamp',
    exclude_words=['svmap', 'svampar'] + mushrooms_to_exclude + baljväxter_to_exclude,
    match_whole_words=True,
    word_match_type='any'
)

# Apply category
food_database_swe.loc[mask_vegetables, 'Category'] = 'Vegetables'

# Create mask for any of the specified 'Gruppering' values
mask_fruits = map_products_to_category(
    df=food_database_swe,
    grouping=['Frukt, bär', 'Nötter, frön', 'Sylt, marmelad, gelé, chutney']
)

# Apply category
food_database_swe.loc[mask_fruits, 'Category'] = 'Fruits'

# Block 1: Bröd group with exact match on whole word 'kex'
mask_pastries_bröd_whole = map_products_to_category(
    df=food_database_swe,
    grouping='Bröd',
    include_words=['kex'],
    match_whole_words=True,
    word_match_type='any'
)

# Block 2: Bröd group with substring match on 'kex', 'deg', 'kaka'
mask_pastries_bröd_partial = map_products_to_category(
    df=food_database_swe,
    grouping='Bröd',
    include_words=['kex', 'deg', 'kaka'],
    match_whole_words=False,
    word_match_type='any'
)

# Block 3: Bullar, kakor, tårtor group with complex conditions
mask_pastries_group = food_database_swe['Gruppering'] == 'Bullar, kakor, tårtor'

# Sub-condition A: any of several sub-words
cond_a = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    include_words=['tårta', 'fyllning', 'gräddad', 'kex', 'bulle', 'rulle', 'bröd', 'bakelse', 'våffla'],
    match_whole_words=False,
    word_match_type='any'
)

# Sub-condition B: all whole words 'mjuk', 'kaka'
cond_b = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    include_words=['mjuk', 'kaka'],
    match_whole_words=True,
    word_match_type='all'
)

# Sub-condition C: any sub-word (kaka/kakor/muffin/boll) and exclude whole word 'choklad'
cond_c_include = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    include_words=['kaka', 'kakor', 'muffin', 'boll'],
    match_whole_words=False,
    word_match_type='any'
)
cond_c_exclude = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    exclude_words=['choklad'],
    match_whole_words=True,
    word_match_type='any'
)
cond_c = cond_c_include & cond_c_exclude

# Sub-condition D: whole word 'skorpor' but not 'osötade'
cond_d_include = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    include_words=['skorpor'],
    match_whole_words=True,
    word_match_type='any'
)
cond_d_exclude = map_products_to_category(
    df=food_database_swe,
    grouping='Bullar, kakor, tårtor',
    exclude_words=['osötade'],
    match_whole_words=True,
    word_match_type='any'
)
cond_d = cond_d_include & cond_d_exclude

# Combine all Bullar/kakor/tårtor logic
mask_pastries_bkt = cond_a | cond_b | cond_c | cond_d

# Combine all pastry-related logic
mask_pastries = mask_pastries_bröd_whole | mask_pastries_bröd_partial | mask_pastries_bkt

# Apply assignment
food_database_swe.loc[mask_pastries, 'Category'] = 'Pastries'

# Manually assign known pastry items
assign_multiple_single_values(
    [
        'Pirog u. fyllning gräddad', 'Mandelkubb', 'Munk fylld m. äppelmos vaniljkräm',
        'Jitterbugg m. mördeg maräng', 'Mandelbiskvi', 'Baklava ', 'Pepparkaksdeg kylvara',
        'Munk friterad m. socker typ somalisk bur saliid'
    ],
    'Pastries'
)

# Block 1: Glass without 'soja' or 'havre' (substring exclusion)
mask_glass_sweets = map_products_to_category(
    df=food_database_swe,
    grouping='Glass',
    exclude_words=['soja', 'havre'],
    match_whole_words=False,
    word_match_type='any'
)

# Block 2: Dryck or Godis with 'choklad' in name (substring)
mask_dryck_choklad = map_products_to_category(
    df=food_database_swe,
    grouping='Dryck',
    include_words=['choklad'],
    match_whole_words=False,
    word_match_type='any'
)

mask_godis_choklad = map_products_to_category(
    df=food_database_swe,
    grouping='Godis',
    include_words=['choklad'],
    match_whole_words=False,
    word_match_type='any'
)

# Combine all sugar and sweets logic
mask_sugar_sweets = mask_glass_sweets | mask_dryck_choklad | mask_godis_choklad

# Apply category
food_database_swe.loc[mask_sugar_sweets, 'Category'] = 'Sugar and sweets'

# Collect additional uncategorized 'Godis' items
additional_godis_items = food_database_swe[
    (food_database_swe['Gruppering'] == 'Godis') &
    (food_database_swe['Category'].isna())
]['Livsmedelsnamn'].tolist()

# Manually assign known sugar and sweets items
assign_multiple_single_values(
    ['Brownie', 'Maräng', 'Toscaglasyr', 'Knäck'] + additional_godis_items,
    'Sugar and sweets'
)

food_database_swe['Category'] = food_database_swe['Category'].fillna(food_database_swe['Gruppering'])

# We remove some categories
food_database_swe = food_database_swe[food_database_swe['Category'] != 'Processed but unassigned']

# Percent-valued rows are excluded because the optimizer expects converted
# mass-based contaminant values.
df_toxins = df_toxins[~df_toxins['Unit'].isin(['%'])]

# Set up a lookup table from df_toxins
toxins_lookup = df_toxins.set_index('Compound type')

compound_types = toxins_lookup.index.tolist()

# We add toxins to the database based on category.
for compound_type in compound_types:
    food_database_swe[compound_type] = food_database_swe['Category'].map(
        lambda cat: toxins_lookup.at[compound_type, cat] if cat in toxins_lookup.columns else None
    )

food_database_swe = food_database_swe.rename(columns={
                                  'Energi': 'kcal',
                                  'Folat, totalt': 'Folat',
                                  'Fosfor, P': 'Fosfor',
                                  'Jod, I': 'Jod',
                                  'Järn, Fe': 'Järn',
                                  'Kalcium, Ca': 'Kalcium',
                                  'Kalium, K': 'Kalium',
                                  'Magnesium, Mg': 'Magnesium',
                                  'Natrium, Na': 'Natrium',
                                  'Salt, NaCl': 'Salt',
                                  'Selen, Se': 'Selen',
                                  'Zink, Zn': 'Zink'})

# Save the file.
filename = 'df_foods_and_toxins_database.csv'
full_output_path = DATA_DIR / filename
food_database_swe.to_csv(full_output_path, index=False)
