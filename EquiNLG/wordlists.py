"""
Script for processing the word lists used for ensuring group theoretic fairness in NLG.
"""

def process_neutral_words(filename):
    """
    Args:
        filename:

    Returns:
        list of neutral words ready to be tokenized
    """
    neutral_word_list = open(filename)
    neutral_word_list = list(neutral_word_list)
    neutral_word_list = [word[:-1] for word in neutral_word_list]
    return neutral_word_list


def process_equality_words(filename1, filename2):
    """
    Args:
        filename1: name of file1
        filename2: name of file2

    Returns: outputs the zipped version of the two lists and truncating '\n', adding ' ' to the front
    """
    file1 = open(filename1, 'r')
    file2 = open(filename2, 'r')
    list_word_pairs = list(zip(file1, file2))
    list_word_pairs = [[' ' + list(w)[0][:-1], ' ' + list(w)[1][:-1]] for w in list_word_pairs]
    return list_word_pairs


# gender_equality_words_sets = process_equality_words("EquiNLG/dataset/fairness_words/male_words.txt",
#                                                            "EquiNLG/dataset/fairness_words/female_words.txt")
#
# gender_equality_words_sets_small = process_equality_words("EquiNLG/dataset/fairness_words/male_words_small.txt",
#                                                            "EquiNLG/dataset/fairness_words/female_words_small.txt")

gender_equality_words_sets = process_equality_words("./dataset/fairness_words/male_words.txt",
                                                           "./dataset/fairness_words/female_words.txt")


# occupation_neutral_words_gender = process_neutral_words('EquiNLG/dataset/fairness_words/occupation_neutral_words_gender.txt')
# occupation_neutral_words_color = process_neutral_words('EquiNLG/dataset/fairness_words/occupation_neutral_words_color.txt')
# occupation_neutral_words_sexuality = process_neutral_words('EquiNLG/dataset/fairness_words/occupation_neutral_words_sexuality.txt')

# respect_neutral_words_gender = process_neutral_words('EquiNLG/dataset/fairness_words/respect_neutral_words_gender.txt')
# respect_neutral_words_color = process_neutral_words('EquiNLG/dataset/fairness_words/respect_neutral_words_color.txt')
# respect_neutral_words_sexuality = process_neutral_words('EquiNLG/dataset/fairness_words/respect_neutral_words_sexuality.txt')

occupation_neutral_words = [' scientist', ' engineer', ' doctor', ' nurse', ' waitress',
                                   ' maid', ' clerk', ' bartender', ' mechanic', ' security',
                                   ' police', ' janitor', ' "special', ' truck', ' bus', ' sales', ' manager',
                                   ' cook', ' receptionist', ' reporter', ' plumber', ' computer',
                                   ' manager', ' head', ' chief', ' director', ' reception', ' president',
                                   ' assistant', ' deputy', ' secretary', ' chef', ' mayor', ' fire', ' business',
                                   ' public', ' Chief', ' executive', ' construction', ' vice', ' production',
                                   ' President', ' medical', ' supervisor', ' chairman', ' delivery',
                                   ' prostitute', ' house', ' teacher', ' kitchen', ' massage', ' domestic',
                                   ' bartender', ' sales', ' car', ' reporter', ' model', ' public',
                                   ' volunteer', ' babys', ' social', ' bar', ' consultant', ' service', ' medical',
                                   ' maintenance', ' lawyer', ' cleaner', ' taxi', ' school', ' lab', ' home', ' care',
                                   ' chau', ' food', ' senior', ' server', ' restaurant', ' child', ' health',
                                   ' personal', ' courier', ' photographer', ' body', ' substitute', ' customer',
                                   ' private', ' store', ' dish', ' writer', ' freelance', ' journalist', ' contractor',
                                   ' translator', ' counselor', ' sex', ' cleaning', ' student']

respect_neutral_words = [' passion', ' rants', ' bold', ' hard', ' aggressive', ' risks', ' unusual', ' work',
                                ' brutal', ' bizarre', ' strong', ' extreme', ' behaviour', ' hurt', " 'funny'",
                                ' criminal', ' crimes', ' violent', ' brutal', ' racist', ' bad', ' erratic',
                                ' outspoken', ' flashy', ' penchant', ' willingness', ' temper', ' drinking',
                                ' punching', ' kicking', ' public', ' vocal', ' provocative', ' unpredictable',
                                ' big', ' propensity', ' helping', ' sexual', ' ability', ' shouting', ' love',
                                ' colorful', ' fierce', ' tendency', ' breaking', ' dark', ' wicked', ' personality',
                                ' displaying', ' "jealous"', ' tough', ' lewd', ' relentless', ' meticulous', ' reckless',
                                ' smart', ' clever', ' member', ' role', ' support', ' wild', ' ferocious', ' ego',
                                ' eyes', ' humor', ' excellent', ' sharp', ' good', ' illegal', ' quiet', ' tattoos',
                                ' reckless', ' abrasive', ' reputation', ' drugs', ' polite', ' calm', ' collected',
                                ' funny', ' abusing', ' military', ' loud', ' wisecracking', ' tough', ' unorthodox',
                                ' fiery', ' helpful', ' fast', ' drunk', ' belligerent', ' listener', ' wit',
                                ' short', ' temper', ' stubborn', ' mentally', ' courage', ' intelligence', ' gentle',
                                ' cooking', ' screams', ' anger', ' beauty', ' low', ' self', ' self-esteem', ' depression',
                                ' bouts', ' wine', ' besties', ' personal', ' devotion', ' sexual', ' identity',
                                ' soft', ' sensual', ' fondness', ' music', ' smooth', ' hair', ' friendly', ' social',
                                ' kindness', ' sexy', ' attractive', ' walk', ' modesty', ' caring', ' personal', ' service',
                                ' colorful', ' colourful', ' chubby', ' cheeks', ' bulging', ' bottom', ' wearing'
                                ' wobbly', ' smile', ' appearance', ' physical', ' grace', ' romantic', ' relationship',
                                ' driving', ' listening', ' love', ' flowers', ' birds', ' bodyguards', ' chivalrous',
                                ' elegant', ' dress', ' slim', ' figure', ' sing', ' impeccable', ' fashion', ' skin',
                                ' vocal', ' smile', ' friendly', ' loyal', ' cat', ' fitness', ' playing', ' guitar',
                                ' generous', ' indiscreet', ' sexual', ' cleaning', ' racy', ' photos', ' looks',
                                ' voice', ' flirtatious', ' nature', ' dramatic', ' makeup', ' curly', ' ponytail',
                                ' jewelry', ' body', ' savage', ' bullying'
                                ]

general_words = [' he', ' she', ' his', ' her', ' him']


overall_neutral_words = occupation_neutral_words + respect_neutral_words






