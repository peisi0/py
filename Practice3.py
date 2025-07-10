import math
import unicodedata

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
    


rad = input("Please enter the radius of cylinder: ")
if is_number(rad):
    hei = input("Please enter the height of cylinder: ")
    if is_number(hei):
        vol = math.pow(int(rad), 2) * math.pi * int(hei)
        tsa = (math.pow(int(rad), 2) * math.pi * 2) + (math.pi * int(rad) * 2 * int(hei)) 
        print("\n===Caculation Process===")
        print("Volume =", end = " ")
        print(vol)
        print("Total surface area =", end = " ")
        print(tsa)
    else:
        print("Please enter the  positive number.")
else:
    print("Please enter the  positive number.")

    
