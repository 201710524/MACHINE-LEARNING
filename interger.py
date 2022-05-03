

list_input_a = [1,2,3,4,5]


output_a= map(lambda x: x*x, list_input_a)
print(output_a)
print(list(output_a))
print()

output_b = filter(lambda x: x <3, list_input_a)
print(output_b)
print(list(output_b))

