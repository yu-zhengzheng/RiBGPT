import os


def assemble(model):
    """
    Assembles the model from part files

    :param model: a string representing model name/directory
    :return: a message string
    """
    file_path = "models/" + model + "/pytorch_model.bin"

    if not os.path.isfile(file_path):

        # Open the input file in binary mode
        output_file = open(file_path, "wb")

        # Initialize the chunk number
        chunk_number = 1

        # Loop until the end of the file is reached
        while True:
            # Create the output file name based on the chunk number
            part_file_path = file_path + f"_part{chunk_number:03d}.bin"
            print("reading file", part_file_path)

            try:
                # Open the output file in binary mode
                part_file = open(part_file_path, "rb")
            except:
                break

            # Read a chunk of data from the input file
            chunk_data = part_file.read()

            # Write the chunk data to the output file
            output_file.write(chunk_data)

            # Close the output file
            part_file.close()

            # Increment the chunk number
            chunk_number += 1

        # Close the input file
        output_file.close()

        return "assembly finished!"

    else:
        return "already assembled!"


if __name__ == '__main__':
    msg=assemble("ft3")
    print(msg)
