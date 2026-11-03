def format_number(n: int) -> str:
    """Format number using K (thousands), M (millions), B (billions).

    Uses X.Y format if there is one digit before the decimal point, otherwise drops decimals.
    """
    if n >= 1_000_000_000:
        val = n / 1_000_000_000
        unit = "B"
    elif n >= 1_000_000:
        val = n / 1_000_000
        unit = "M"
    elif n >= 1_000:
        val = n / 1_000
        unit = "K"
    else:
        return str(n)

    if 1.0 <= round(val, 1) < 10.0:
        return f"{val:.1f}{unit}"
    else:
        return f"{round(val):.0f}{unit}"
