CLASS_NAMES = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Motorbike",
    4: "Bicycle",
    5: "Rickshaw"
}


model.names = CLASS_NAMES


assert len(model.names) == NUM_CLASSES, "Class mismatch with dataset"
