#####Loading data back in#########
# Specify our path
PATH = "model.pt"

# Create a new "blank" model to load our information into
model = FirstNet()

# Recreate our optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load back all of our data from the file
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']