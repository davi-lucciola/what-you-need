from langchain.tools import tool


type number = int | float


@tool
def add(a: number, b: number) -> number:
    """Add a + b and returns the result

    Args:
        a: number addend
        b: number addend

    Returns:
        the resulting float of the equation a + b
    """
    return a + b


@tool
def subtract(a: number, b: number) -> number:
    """Subtract a - b and returns the result

    Args:
        a: number minuend
        b: number subtrahend

    Returns:
        the resulting float of the equation a - b
    """
    return a - b


@tool
def multiply(a: number, b: number) -> number:
    """Multiply a * b and returns the result

    Args:
        a: number mutiplicant
        b: number multiplier

    Returns:
        the resulting float of the equation a * b
    """
    return a * b
