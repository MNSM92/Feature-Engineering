def P(text, color):
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }

    if color not in color_codes:
        print("Invalid color. Please choose one of: red, green, yellow, blue, purple, cyan, white.")
        return

    print(f"{color_codes[color]}{text}{color_codes['reset']}")
