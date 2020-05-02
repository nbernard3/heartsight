
from datetime import datetime

def timestamp(name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), name)

if __name__ == '__main__':
    pass
