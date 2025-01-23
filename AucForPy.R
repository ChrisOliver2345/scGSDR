load_matrix <- function(path){
  expr_matrix = read.table(path,sep=',',header=T,row.names=1)
  expr_matrix = as.matrix(expr_matrix)
  return(expr_matrix)
}

load_pathway <- function(path){
  gSet=getGmt(path)
  return(gSet)
}

pathway_scoring <- function(gSet, mat_gene){
  cells_rankings <- AUCell_buildRankings(mat_gene, plotStats=FALSE)
  cells_AUC <- AUCell_calcAUC(gSet, cells_rankings, nCores=1,aucMaxRank = 300)
  aucMatrix <- getAUC(cells_AUC)
  aucMatrix = aucMatrix[rowSums(aucMatrix)>0.0,]
  return(aucMatrix)
}


AUC <- function(scPath, dataset, output_paths, refresh) {

    mat_gene = load_matrix(scPath) # genes*cells

    if (dataset=='Reactome_Human+PharmGKB'){
        output_path_meta = output_paths[[1]]
        output_path_cp = output_paths[[2]]
        output_path_eip = output_paths[[3]]
        output_path_os = output_paths[[4]]
        output_path_di = output_paths[[5]]
        output_path_dr = output_paths[[6]]
    }

    if (dataset=='KEGG_Human+PharmGKB'){
        output_path_meta = output_paths[[1]]
        output_path_cp = output_paths[[2]]
        output_path_eip = output_paths[[3]]
        output_path_os = output_paths[[4]]
        output_path_di = output_paths[[5]]
        output_path_dr = output_paths[[6]]
        output_path_ot = output_paths[[7]]
    }

    if (dataset=='Reactome_Human'){
        output_path_meta = output_paths[[1]]
        output_path_cp = output_paths[[2]]
        output_path_eip = output_paths[[3]]
        output_path_os = output_paths[[4]]
        output_path_di = output_paths[[5]]
    }

    if (dataset=='KEGG_Human'){
        output_path_meta = output_paths[[1]]
        output_path_cp = output_paths[[2]]
        output_path_eip = output_paths[[3]]
        output_path_os = output_paths[[4]]
        output_path_di = output_paths[[5]]
        output_path_ot = output_paths[[6]]
    }

    pathway_meta = "./Pathways/Human_Metabolism.txt"
    pathway_cp = "./Pathways/Human_Cellular_Process.txt"
    pathway_eip = "./Pathways/Human_Environmental_Information_Processing.txt"
    pathway_os = "./Pathways/Human_Organismal_Systems.txt"
    pathway_di = "./Pathways/Human_Disease.txt"
    pathway_dr = "./Pathways/Human_PharmGKB_Drug.txt"

    if (!file.exists(output_path_meta) || refresh == 1){
        gSet=load_pathway(pathway_meta)
        gSet=subsetGeneSets(gSet, rownames(mat_gene))
        print(gSet)
        mat_path = pathway_scoring(gSet, mat_gene)
        write.csv(mat_path,file=output_path_meta)
    }
    else{
      print("Metabolism scored File exists.")
    }

    if (!file.exists(output_path_cp)|| refresh == 1){
      gSet=load_pathway(pathway_cp)
      gSet=subsetGeneSets(gSet, rownames(mat_gene))
      print(gSet)
      mat_path = pathway_scoring(gSet, mat_gene)
      write.csv(mat_path,file=output_path_cp)
    }
    else{
      print("Cellular_Process scored File exists.")
    }

    if (!file.exists(output_path_eip)|| refresh == 1){
      gSet=load_pathway(pathway_eip)
      gSet=subsetGeneSets(gSet, rownames(mat_gene))
      print(gSet)
      mat_path = pathway_scoring(gSet, mat_gene)
      write.csv(mat_path,file=output_path_eip)
    }
    else{
      print("Environmental_Information_Processing scored File exists.")
    }

    if (!file.exists(output_path_os)|| refresh == 1){
      gSet=load_pathway(pathway_os)
      gSet=subsetGeneSets(gSet, rownames(mat_gene))
      print(gSet)
      mat_path = pathway_scoring(gSet, mat_gene)
      write.csv(mat_path,file=output_path_os)
    }
    else{
      print("Organismal_Systems scored File exists.")
    }

    if (!file.exists(output_path_di)|| refresh == 1){
      gSet=load_pathway(pathway_di)
      gSet=subsetGeneSets(gSet, rownames(mat_gene))
      print(gSet)
      mat_path = pathway_scoring(gSet, mat_gene)
      write.csv(mat_path,file=output_path_di)
    }
    else{
      print("Disease scored File exists.")
    }

    if (dataset=='Reactome_Human+PharmGKB' | dataset=='KEGG_Human+PharmGKB'){
        if (!file.exists(output_path_dr)|| refresh == 1){
          gSet=load_pathway(pathway_dr)
          gSet=subsetGeneSets(gSet, rownames(mat_gene))
          print(gSet)
          mat_path = pathway_scoring(gSet, mat_gene)
          write.csv(mat_path,file=output_path_dr)
        }
        else{
          print("Drug scored File exists.")
        }
    }

    if (dataset=='KEGG_Human+PharmGKB' | dataset=='KEGG_Human'){
        if (!file.exists(output_path_ot)|| refresh == 1){
          gSet=load_pathway(pathway_ot)
          gSet=subsetGeneSets(gSet, rownames(mat_gene))
          print(gSet)
          mat_path = pathway_scoring(gSet, mat_gene)
          write.csv(mat_path,file=output_path_ot)
        }
        else{
          print("Other scored File exists.")
        }
    }

}

