def merge_tree(sent):
	result = ""
	for line in sent:
		openb = line[2].translate(None,"1234567890)_|")
		closeb = line[2].translate(None,"1234567890(_|")
		for x in range(len(openb)):
			result += "Tree('NP',[" 
		result += ("," if not openb else "") + "('" + line[0] + "','" + line[1] +  "')"
		for x in range(len(closeb)):
			result += "])"
	return result