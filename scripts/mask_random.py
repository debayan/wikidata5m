import sys,os,json,random

def mask_words_with_percentage(text, mask_percentage=0.15):
    # Split the input string into words
    words = text.split()

    # Calculate the number of words to mask
    num_words_to_mask = max(1, int(len(words) * mask_percentage))

    # Randomly select the positions to mask
    masked_positions = random.sample(range(len(words)), num_words_to_mask)

    # Replace the selected words with the [MASK] token
    for pos in masked_positions:
        words[pos] = "[MASK]"

    # Join the words back together to form the masked text
    masked_text = " ".join(words)

    return masked_text

f = open('../data/mask-pre-train-data.txt','w')
with open('../data/pre-train-data.txt') as infile:
    for line in infile:
        line = line.strip()
        masked_line = mask_words_with_percentage(line)
        #print(line)
        #print(masked_line)
        #print("------------")
        f.write(masked_line+'\n')
f.close()

  
