import json
import tiktoken
import os

def estimate_tokens():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "data", "locomo10.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    encoding = tiktoken.get_encoding("o200k_base") # common encoding
    
    sample = data[0]
    if 'conversation' in sample:
        sample_conv = sample['conversation']
    else:
        sample_conv = sample
        
    total_tokens = 0
    
    session_nums = []
    for k in sample_conv.keys():
        if k.startswith('session_') and not k.endswith('_date_time'):
            parts = k.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                session_nums.append(int(parts[1]))
                
    num_sessions = len(session_nums)
    
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in sample_conv:
            session_date = sample_conv.get('session_%s_date_time' % i, "")
            header = 'DATE: ' + session_date + '\n' + 'CONVERSATION:\n'
            total_tokens += len(encoding.encode(header))
            
            for dialog in sample_conv['session_%s' % i]:
                turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
                total_tokens += len(encoding.encode(turn))
                
    print(f"Sample 0: {num_sessions} sessions, Total Tokens: {total_tokens}")

    # Estimate for all samples
    totals = []
    for idx, sample in enumerate(data):
        if 'conversation' in sample:
            sample_conv = sample['conversation']
        else:
            sample_conv = sample
            
        total_tokens = 0
        session_nums = []
        for k in sample_conv.keys():
            if k.startswith('session_') and not k.endswith('_date_time'):
                parts = k.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    session_nums.append(int(parts[1]))
        
        if not session_nums:
            continue

        for i in range(min(session_nums), max(session_nums) + 1):
            if 'session_%s' % i in sample_conv:
                session_date = sample_conv.get('session_%s_date_time' % i, "")
                header = 'DATE: ' + session_date + '\n' + 'CONVERSATION:\n'
                total_tokens += len(encoding.encode(header))
                for dialog in sample_conv['session_%s' % i]:
                    turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"\n'
                    if "blip_caption" in dialog:
                        turn += ' and shared %s.' % dialog["blip_caption"]
                    turn += '\n'
                    total_tokens += len(encoding.encode(turn))
        totals.append(total_tokens)
    
    print(f"Max tokens across all samples: {max(totals)}")
    print(f"Average tokens across all samples: {sum(totals)/len(totals)}")

if __name__ == "__main__":
    estimate_tokens()
