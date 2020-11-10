def addressbook(name_to_phone: dict, name_to_address: dict):
    address_to_all = dict() #key: addrss, val: (name, phone)
    
    for name, address in name_to_address.items():
        if address in address_to_all.keys(): #duplicate, want to add more names
            temp1 = address_to_all.get(address)[0]
            temp1.append(name)
            address_to_all[address] = (temp1, address_to_all[address][1])
        else: #first address, get name and phone
            address_to_all[address] = ([name], str(name_to_phone.get(name)))

    #before returning, check if all person at the address uses the same phone number
    for k, v in address_to_all.items():
        if len(v[0]) > 1: #check if there's multiple people using same address
            first_phone = name_to_phone.get(v[0][0])
            diff = set()
            for name in v[0][1:]:
                if str(name_to_phone.get(name)) !=  str(first_phone):
                    diff.add(name)
            if len(diff) != 0:
                diff = str(diff).replace('{', '')
                diff = str(diff).replace('}', '')                
                print("Warning: " + diff.replace("'","") + " has a different number for " + k + " than " + str(v[0][0]) + ". Using the number for " + str(v[0][0]) + " .")
                
    return address_to_all

if __name__ == "__main__":
    name_to_phone = {'alice': 5678982231, 'bob': '111-234-5678', 'christine': 5556412237, 'daniel': '959-201-3198', 'edward': 5678982232}
    name_to_address = {'alice': '11 hillview ave', 'bob': '25 arbor way', 'christine': '11 hillview ave', 'daniel': '180 ways court', 'edward': '11 hillview ave'}
    address_to_all = addressbook(name_to_phone, name_to_address)
    print(address_to_all)