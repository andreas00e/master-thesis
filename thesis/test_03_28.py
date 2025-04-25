from tqdm import tqdm
import multiprocessing as mp

def process_item(item):
    """Processes an item and returns it (simulating work)."""
    return item  # You can modify this function to do actual work

if __name__ == "__main__":
    large_list = list(range(10000))  # Example large list

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_item, large_list)):
            continue
            print(result)  # Process each result as it comes in
