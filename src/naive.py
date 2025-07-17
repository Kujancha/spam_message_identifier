import math
from collections import defaultdict



        
def tokenize(text:str):
    return text.lower().split()  # .split makes it so that the messages are turned into a list

class NaiveBayesClassifier:
    def __init__(self):
        self.spam_word_count = defaultdict(int)  # dictionary to hold the spam word count (if multiple exists the numeric vcalue key will also increase)
        self.ham_word_count = defaultdict(int)
        
        self.vocab = set()  #this is to store a set of list of words in  a message
        
        self.num_spam = 0    # this two to store the numerical ampount of spam and ham messages
        self.num_ham = 0
        
        self.spam_total = 0   # this is to store the numeric value of words in spam or ham
        self.ham_total = 0
           
    def train(self, data:list):
        for message, label in data:
            words = tokenize(message)
            self.vocab.update(words)
            
            if label.lower() == "spam":
                self.num_spam +=1
                
                for word in words:
                    self.spam_word_count[word] +=1
                    self.spam_total +=1
            
            else:
                self.num_ham +=1
                
                for word in words:
                    self.ham_word_count[word] +=1
                    self.ham_total +=1
                    
        
        self.total_messages = self.num_spam + self.num_ham
        self.vocab_size = len(self.vocab) 
    
    def predict(self, message):
        words = tokenize(message)
        
        #using logarithmic probablities to avoid underflow lol
        
        log_spam = math.log(self.num_spam / self.total_messages)
        log_ham = math.log(self.num_ham / self.total_messages)
        
        for word in words:
            
            # this is called a Laplce smoothing, helps in case where the prob comes out as zero
            
            p_word_given_spam = (self.spam_word_count[word] + 1) / (self.spam_total + self.vocab_size)  
            p_word_given_ham = (self.ham_word_count[word] + 1) / (self.ham_total + self.vocab_size)
            
            log_spam += math.log(p_word_given_spam)
            log_ham += math.log(p_word_given_ham)
            
        return "spam" if log_spam > log_ham else "ham"
    
    
def main():
    # train set      
    train_data = [
        ("Win a free cruise now", "spam"),
        ("Limited time offer, buy now", "spam"),
        ("Meet me for lunch today", "ham"),
        ("Your appointment is scheduled for tomorrow", "ham"),
        ("Less have ice-cream at 5 today sweetie", "ham"),
        ("Congratulations you just won an Iphone 17, Click here to redeem now", "spam"),
        ("You won't believe what she says next, click here to find more now", "spam"),
        ("mum wants to talk to you now", "ham")
    ]

  
  
    trainset = NaiveBayesClassifier() 
    trainset.train(train_data)
    
    test_messages = [
    "Free lunch offer now",     # should be spam
    "Dad's calling me now, i am worried",   # should be ham
    "Win money now",            # should be spam
    "Schedule lunch today"      # should be ham
    ]

    for msg in test_messages:
        prediction = trainset.predict(msg)
        print(f"Message: '{msg}' → Prediction: {prediction}")
        
    
    
if __name__ == "__main__":
    main()
    pass