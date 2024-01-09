import re

class WikiDataProcessor:
   def __init__(self, file_path):
      self.file_path = file_path

   def read_line_by_line(self):
      with open(self.file_path, 'r', encoding='utf-8') as file:
         for line in file:
            yield line.strip()

   def read_word_by_word(self):
      with open(self.file_path, 'r', encoding='utf-8') as file:
         for line in file:
            words = re.findall(r'\b\w+\b', line.lower())
            for word in words:
               yield word

   def read_sentence_by_sentence(self):
      with open(self.file_path, 'r', encoding='utf-8') as file:
         text = file.read()
         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
         for sentence in sentences:
            yield sentence.strip()

# # Example usage
# file_path = '/lus/eagle/projects/atlas_aesp/data/wikitext-103-raw/wiki.valid.raw'
# processor = WikiDataProcessor(file_path)

# # Read line by line
# for line in processor.read_line_by_line():
#    print(line)

# # Read word by word
# for word in processor.read_word_by_word():
#    print(word)

# # Read sentence by sentence
# for sentence in processor.read_sentence_by_sentence():
#    print(sentence)
