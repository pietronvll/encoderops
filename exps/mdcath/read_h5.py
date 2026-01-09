import h5py


def print_keys_recursively(group, call_stack=0):
    if hasattr(group, "keys"):
        for key in group.keys():
            print("-" * call_stack + key)
            print_keys_recursively(group[key], call_stack=call_stack + 1)


if __name__ == "__main__":
    file_path = "/home/novelli/encoderops/datasets/mdcath/mdcath_dataset_1a1zA00.h5"
    with h5py.File(file_path, "r") as f:
        # Print all groups recursively
        print_keys_recursively(f)
