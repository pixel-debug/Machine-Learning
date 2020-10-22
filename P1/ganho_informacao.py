

def entropia(df_dados,nom_col_classe):
   
    ser_count_col = df_dados[nom_col_classe].value_counts()
    num_total = len(df_dados)
    entropia = 0
 
    for val,count_atr in ser_count_col.iteritems():   
        aux_t = count_atr / num_total
        
        
        if aux_t == 0 :
            entropia = 0
            break
        entropia += count_atr / num_total * (-(math.log(aux_t,2)) )
        
        val_prob = 0
        entropia += 0
       
    return entropia


def ganho_informacao_condicional(df_dados,val_entropia_y,nom_col_classe,nom_atributo,val_atributo):
   
    val_gi = 0
    val_entropia = 0
    
    df_dados_filtrado = df_dados[df_dados[nom_atributo] == val_atributo]
   
    entropia_y = entropia(df_dados_filtrado, nom_col_classe)
    
    val_gi = val_entropia_y - entropia_y

    return val_gi


def ganho_informacao(df_dados,nom_col_classe,nom_atributo):
   
    num_total = len(df_dados)
    info_gain = 0
    ser_count_col = df_dados[nom_atributo].value_counts()
    for val,count_atr in ser_count_col.iteritems():
        
        media =(count_atr/(num_total))
        gain = ganho_informacao_condicional(df_dados,entropia(df_dados, nom_col_classe),nom_col_classe,nom_atributo,val) 
       
        info_gain = info_gain + (media)*(gain)
    return info_gain