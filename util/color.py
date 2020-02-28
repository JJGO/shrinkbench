"""Auxiliary Module for color printing in terminal and jupyter notebooks
"""


class colors:
    END          = "\033[0m"
    RESET        = "\033[0m"
    BOLD         = "\033[1m"
    UNDERLINE    = "\033[4m"
    REVERSED     = "\033[7m"
    BLACK        = "\033[30m"
    FADERED      = "\033[31m"
    GRASS        = "\033[32m"
    YELLOW       = "\033[33m"
    NAVY         = "\033[34m"
    PURPLE       = "\033[35m"
    DARKCYAN     = "\033[36m"
    WHITE        = "\033[37m"
    GREY         = "\033[90m"
    RED          = "\033[91m"
    GREEN        = "\033[92m"
    ORANGE       = "\033[93m"
    BLUE         = "\033[94m"
    MAGENTA      = "\033[95m"
    CYAN         = "\033[96m"
    BRIGHT       = "\033[97m"
    BG_BLACK        = "\033[90m"
    BG_FADERED      = "\033[91m"
    BG_GRASS        = "\033[92m"
    BG_YELLOW       = "\033[93m"
    BG_NAVY         = "\033[94m"
    BG_PURPLE       = "\033[95m"
    BG_DARKCYAN     = "\033[96m"
    BG_WHITE        = "\033[97m"
    BG_GREY         = "\033[100m"
    BG_RED          = "\033[101m"
    BG_GREEN        = "\033[102m"
    BG_ORANGE       = "\033[103m"
    BG_BLUE         = "\033[104m"
    BG_MAGENTA      = "\033[105m"
    BG_CYAN         = "\033[106m"
    BG_BRIGHT       = "\033[107m"


def _color2code(color, bg=False):
    """Converts from color formats to ASCII sequence

    Arguments:
        color -- Color to use, color can be one of:
        - (R,G,B) - tuple/list of ints in range 0-255
        - #XXXXXX - String with hex representation of color. Parsed to tuple
        - RGB(X,Y,Z) - String with this format. Parsed to tuple
        - name - String with a name of an attribute of colors (see above)

    Keyword Arguments:
        bg {bool} -- Whether this is a background color (default: {False})

    Returns:
        str -- ANSI color escape sequence
    """

    if isinstance(color, str):
        color = color.upper()
        if hasattr(colors, color):
            if bg:
                color = "BG_" + color
            return getattr(colors, color)
        elif color.startswith("#"):
            r, g, b = color[1:3], color[3:5], color[5:7]
            rgb = [int(x, 16) for x in (r, g, b)]
            return _color2code(rgb, bg)
        if color.startswith("RGB("):
            rgb = [int(x) for x in color[4:-1].split(',')]
            return _color2code(rgb, bg)
    elif isinstance(color, (list, tuple)):
        assert len(color) == 3, "For tuple input length must be 3"
        code = "\033[38;2;" if not bg else "\033[48;2;"
        r, g, b = color
        code += f"{r};{g};{b}m"
        return code


def highlight(text, match, color='BRIGHT', bg=None):
    """Highlight an exact match of text

    Arguments:
        text {str} -- Text to search through
        match {str} -- Match to highlight

    Keyword Arguments:
        color {str} -- [description] (default: {'YELLOW'})

    Returns:
        str -- Text with highlighted match
    """
    prefix = _color2code(color)
    if bg is not None:
        prefix += _color2code(bg, bg=True)
    return text.replace(match, prefix+match+colors.END)


def printc(*args, color='BOLD', bg=None, **kwargs):
    """Print with color

    [description]

    Arguments:
        *args, **kwargs -- Arguments to pass to print

    Keyword Arguments:
        color -- Foreground color to use (default: {'BOLD'})
        bg --  Background color to use (default: {None})
    """
    prefix = _color2code(color)
    if bg is not None:
        prefix += _color2code(bg, bg=True)
    print(prefix, end='')
    print(*args, **kwargs)
    print(colors.END, end='')


if __name__ == '__main__':
    printc("hello RED", color='bold')
    printc("hello RED", color='RED')
    printc("hello RED", color='#CC0000')
    printc("hello RED", color='#992233')
    printc("hello RED", color='rgb(200,200,100)')
    printc("hello RED", color=(200, 200, 100))
    printc(highlight("=> Hello foo, Hello bar, Hello baz", "Hello", color='REVERSED'))

    printc("hello RED", color='NAVY', bg='RED')
    printc("hello RED", color='NAVY', bg='#CC0000')
    printc("hello RED", color='NAVY', bg='#992233')
    printc("hello RED", color='NAVY', bg='rgb(200,200,100)')
    printc("hello RED", color='NAVY', bg=(200, 200, 100))
    printc(highlight("=> Hello foo, Hello bar, Hello baz,", "Hello", color='UNDERLINE', bg='RED'))
