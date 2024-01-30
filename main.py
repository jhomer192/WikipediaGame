
import wikipediaapi
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Method that returns an array of the path taken between two wikipedia pages
def find_target_path(start_title, end_title):
    #Telling wikipedia who I am for their records
    wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='Wikipedia Game Bot (jack@homerfamily.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)
    #Fetching the wikipedia pages (title, text, links and more)
   
    start_doc = wiki_wiki.page(start_title)
    end_doc = wiki_wiki.page(end_title)
    if not start_doc.exists() or not end_doc.exists(): #if doc names entered don't exist
        print("one or both articles entered in don't exist. Terminating the program.")
        sys.exit()
    #make sure the pages exist
    current_doc = start_doc
    result = [start_title] #list of path taken 
    previous_visits = set(start_doc.title) #set that ensures we don't double visit
    print(f"Currently Visiting: {current_doc.title}, has {len(current_doc.links)} links") #sends out first status message
    while current_doc.title != end_doc.title: #since wikipedia pages names are unique we use them to verify
        vectorizer = TfidfVectorizer() #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        adjacent_docs = current_doc.links #list of all wikipedia articles connect to the current one
        if end_doc.title in adjacent_docs.keys(): #if we find the end doc here we know that we connected the path
            break
        fitting_list = [x.text for x in adjacent_docs.values()] #list of all texts of adjacent articles
        vectorizer.fit(fitting_list + [end_doc.text]) #we have to fit all text including target to find the most optimal 
        document_vectors =  vectorizer.transform(fitting_list) #have to transform using td-idf
        end_vector = vectorizer.transform([end_doc.text]) #transform the result using the current model so we can compare it
        relative_scores = [] 
        for document in document_vectors:
            relative_scores.append(cosine_similarity(document, end_vector)) #compare all adjacent articles with the end goal using consine similarity
        to_compare = [(a[0],a[1]) for a in zip(adjacent_docs.keys(), relative_scores)] #zip the titles with the similarity scores 
        to_compare.sort(key = lambda x: -1 * x[1]) #sort by similarity scores descending order
        cur_val = 0
        while to_compare[cur_val][0] in previous_visits: #finds the highest scoring article we haven't visited
            cur_val += 1
        current_doc = wiki_wiki.page(to_compare[cur_val][0]) #gets the doc that we determine is similar
        result.append(current_doc.title) #add new doc to the path
        previous_visits.add(to_compare[cur_val][0]) #adds doc to set so we don't double visit
        previous_visits.add(current_doc.title) #adds title in case its a redirect article
        print(f"Currently Visiting: {current_doc.title}, has {len(current_doc.links)} links. Has a similarity score to {end_doc.title} of {to_compare[0][1]}")
    result.append(end_doc.title) 
    return result
start_title = input("Pick a starting article (name has to match the page): ") #takes the inputs
end_title = input("Pick an ending article (name has to match the page): ")
print(f"Found! Your path was {find_target_path(start_title, end_title)}")