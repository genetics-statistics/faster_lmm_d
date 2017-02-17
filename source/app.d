import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import std.getopt;
import std.json;
import simplelmm.rqtlreader;
import simplelmm.tsvreader;
import simplelmm.lmm;
import simplelmm.gwas;
//import simplelmm.genotype;
//import simplelmm.phenotype;
import simplelmm.dmatrix;
import simplelmm.optmatrix;
import simplelmm.opencl.add;

void main(string[] args)
{
  // Main routine
  //tryAdd();

  dmatrix a = dmatrix([5,5],[   0.67,  0.00,  0.00,  0.00,  0.00,
                               -0.20,  3.82,  0.00,  0.00,  0.00,
                                0.19, -0.13,  3.27,  0.00,  0.00,
                               -1.06,  1.06,  0.11,  5.86,  0.00,
                                0.46, -0.48,  1.10, -0.98,  3.54
                            ]);
  dmatrix b,c;
  eigh(a, b,c);
  pPrint(b);
  writeln(c);

  string ocontrol, okinship, opheno, ogeno, useBLAS, noBLAS, noCUDA;
  int pheno_column;
  string cmd;

  getopt(args, "control", &ocontrol, "kinship", &okinship, "pheno", &opheno, "geno", &ogeno, "useBLAS", &useBLAS, "noBLAS", &noBLAS, "noCUDA", &noCUDA, "pheno_column", &pheno_column, "cmd", &cmd);

  writeln(cmd);
  JSONValue ctrl;

  if(cmd == "rqtl"){
    writeln("import rqtlreader as reader");
  }
  else{
    writeln("import tsvreader as reader");
  }

  if(ocontrol){
    ctrl = control(ocontrol);//type
    writeln(ctrl);
  }

  if(okinship){
    //string k = reader.kinship(kinship);
    //kinship(kinship);
    writeln("k.shape");
  }
  double[] y;
  string[] ynames;

  if(opheno){
    tsvpheno(opheno, y, ynames, pheno_column);
    writeln(y.sizeof);
  }
  dmatrix g;
  string[] gnames;
  if(ogeno && cmd != "iterator"){
    tsvgeno(ogeno, ctrl, g, gnames);
    writeln(g.shape);
  }

  if(useBLAS){
    bool optmatrixUseBLAS = true;
    writeln(optmatrixUseBLAS);
    writeln("Forcing BLAS support");
  }

  if(noBLAS){
    bool optmatrixUseBLAS = false;
    writeln(optmatrixUseBLAS);
    writeln("Disabling BLAS support");
  }

  if(noCUDA){
    bool cudauseCUDA = false;
    writeln("Disabling CUDA support");
  }

  if(cmd){
    writeln("Error: Run command is missing!");
  }

  //geno_callback("data/small.geno");
int n;
int m;
//if(y!=null){
//  n = y[0];//.sizeof;
//}

//lmmoptions.set(options)
//print lmmoptions.get()

//# If there are less phenotypes than strains, reduce the genotype matrix
  if(g.shape[0] != y.sizeof){
    writeln("Reduce geno matrix to match phenotype strains");
    
    //run_gwas();
    writeln("gnames and ynames");
    writeln(gnames);
    writeln(ynames);
    writeln("gnames and ynames");
    int[] gidx = [];
    int index = 0;
    foreach(ind; ynames){
      while(gnames[0] != ind)
      {
        gnames.popFront;
        index++;
      }
      gidx ~= index;
      index++;
      gnames.popFront;
    }
    writeln(gidx);
    //g2 = g.T[(gidx)].T;
    dmatrix gTranspose = matrixTranspose(g);
    //writeln(gTranspose.shape);
    dmatrix slicedMatrix = sliceDmatrix(gTranspose, gidx);
    writeln(slicedMatrix.shape);
    dmatrix g2 = matrixTranspose(slicedMatrix);
    //prettyPrint(g2);
    //prettyPrint(gTranspose);
    writeln("geno matrix ",g.shape," reshaped to ",g2.shape);
    g = g2;
  }

  if(cmd == "run" || cmd == "rqtl"){
    //if options.remove_missing_phenotypes{
    //  raise Exception('Can not use --remove-missing-phenotypes with LMM2')
    //}
    n = cast(int)y.length;
    m = g.shape[1];
    dmatrix k;
    run_gwas("other",n,m,k,y,g);
    //ps = gwas["ps"];
    //ts = gwas["ts"];
    //check_results(ps,ts);
  }
//else if(cmd == "rqtl"){
//    if options.remove_missing_phenotypes:
//        raise Exception('Can not use --remove-missing-phenotypes with R/qtl2 LMM2')
//    n = len(y)
//    m = g.shape[1]
//    ps, ts = run_gwas('other',n,m,k,y,g)  # <--- pass in geno by SNP
//    check_results(ps,ts)
//  }
//  else if(cmd == "iterator"){
//     if options.remove_missing_phenotypes:
//          raise Exception('Can not use --remove-missing-phenotypes with LMM2')
//      geno_iterator =  reader.geno_iter(options.geno)
//      ps, ts = gn2_load_redis_iter('testrun_iter','other',k,y,geno_iterator)
//      check_results(ps,ts)

  else{
    writeln("Doing nothing");
  }


  //int a = pheno("./data/rqtl/iron_covar.csv");
 // writeln(a);
	//writeln("Edit source/app.d to start your project.");


  //Node root = control("./source/input.yaml");
  //writeln("The answer is ", root["crosstype"].as!int);

  //kinship("./data/rqtl/iron_covar.csv");

 // //Dump the loaded document to output.yaml.
 // Dumper("output.yaml").dump(root);
}

unittest{
  double[] y;
  string[] ynames;

  pheno(opheno, y, ynames, pheno_column);
  writeln(y.sizeof);

  dmatrix g;
  string[] gnames;


  // geno tests

  geno(ogeno, ctrl, g, gnames);
  writeln(g.shape);

  G = matrixTranspose(g);
  //G = np.apply_along_axis( genotype.normalize, axis=1, arr=G)
  K = kinship_full(G);

  //print "Genotype",G.shape, "\n", G
  //print "first Kinship method",K.shape,"\n",K
  k1 = round(K[0][0],4);
  //K2,G = calculate_kinship_new(np.copy(G))
  //print "Genotype",G.shape, "\n", G
  //print "GN2 Kinship method",K2.shape,"\n",K2
  k2 = round(K2[0][0],4);

  if(filname==""){
    assert(k1 == 0.8333);
    assert(k2==0.9375);
    assert(k3==0.9375);
  }
  if(filname==""){
    assert(k1 == 0.8333);
    assert(k2 == 0.7172);
    assert(k3 == 0.7172);
  }
  if(filname==""){
    assert(k3 == 1.4352);
  }
}