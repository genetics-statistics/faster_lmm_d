import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import dyaml.all;
import std.getopt;
import std.json;
import simplelmm.rqtlreader;
import simplelmm.lmm;
import simplelmm.gwas;
//import simplelmm.genotype;
//import simplelmm.phenotype;
import simplelmm.dmatrix;
import simplelmm.optmatrix;

void main(string[] args)
{
  // Main routine

  //auto ctrl;  // R/qtl control structure
  //auto k;
  //auto y;
  //auto g;

  dmatrix d = dmatrix([4,3],[1,2,3,4,5,6,7,8,9,10,11,12]);
  dmatrix e = dmatrix([3,4],[2e-1,3,4,5,3,2,2e-1,3,4,5,3,2]);
  writeln(d.shape);
  writeln(d.elements);
  dmatrix z;
  //z = matrixMultT(d,e);
  z = matrixTranspose(d);
  writeln(z.shape);
  writeln(z.elements);
  prettyPrint(d);
  prettyPrint(z);

  string ocontrol;
  string okinship;
  string opheno;
  string ogeno;
  string useBLAS;
  string noBLAS;
  string noCUDA;
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
    writeln(".....///////////////////////////////..................");
    writeln(ctrl);
    writeln(".....///////////////////////////////..................");
  }

  if(okinship){
    //string k = reader.kinship(kinship);
    //kinship(kinship);
    writeln("k.shape");
  }
  double[] y;
  string[] ynames;

  if(opheno){
    pheno(opheno, y, ynames, pheno_column);
    writeln(y.sizeof);
  }
  dmatrix g;
  string[] gnames;
  if(ogeno && cmd != "iterator"){
    geno(ogeno, ctrl, g, gnames);
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
    writeln(gTranspose.shape);
    dmatrix slicedMatrix = sliceDmatrix(gTranspose, gidx);
    writeln(slicedMatrix.shape);
    dmatrix g2 = matrixTranspose(slicedMatrix);
    prettyPrint(g2);
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
//  }
//  else if(cmd == 'redis_new'){
//    //# The main difference between redis_new and redis is that missing
//    //  # phenotypes are handled by the first
//    if options.remove_missing_phenotypes){
//      raise Exception('Can not use --remove-missing-phenotypes with LMM2')
//    }
//    Y = y;
//    G = g;
//    print "Original G",G.shape, "\n", G
//    // gt = G.T
//    // G = None
//    ps, ts = gn2_load_redis('testrun','other',k,Y,G,new_code=True)
//    check_results(ps,ts)
//  }
//  else if(cmd == "redis"){
//    raise Exception("Obsoleted - all normalization actions are now internal to pylmm")
//    //# Emulating the redis setup of GN2
//    G = g
//    print "Original G",G.shape, "\n", G
//    if y is not None and options.remove_missing_phenotypes:
//        gnt = np.array(g).T
//        n,Y,g,keep = phenotype.remove_missing(n,y,gnt)
//        G = g.T
//        print "Removed missing phenotypes",G.shape, "\n", G
//    else:
//        Y = y
//    if options.maf_normalization:
//        G = np.apply_along_axis( genotype.replace_missing_with_MAF, axis=0, arr=g )
//        print "MAF replacements: \n",G
//    if options.genotype_normalization:
//        G = np.apply_along_axis( genotype.normalize, axis=1, arr=G)
//    g = None
//    gnt = None

//    # gt = G.T
//    # G = None
//    ps, ts = gn2_load_redis('testrun','other',k,Y,G, new_code=False)
//    check_results(ps,ts)
//}else if(cmd == 'kinship'){
//  G = g
//    print "Original G",G.shape, "\n", G
//    if y != None and options.remove_missing_phenotypes:
//        gnt = np.array(g).T
//        n,Y,g,keep = phenotype.remove_missing(n,y,g.T)
//        G = g.T
//        print "Removed missing phenotypes",G.shape, "\n", G
//    if options.maf_normalization:
//        G = np.apply_along_axis( genotype.replace_missing_with_MAF, axis=0, arr=g )
//        print "MAF replacements: \n",G
//    if options.genotype_normalization:
//        G = np.apply_along_axis( genotype.normalize, axis=1, arr=G)
//    g = None
//    gnt = None

//    if options.test_kinship:
//        K = kinship_full(np.copy(G))
//        print "Genotype",G.shape, "\n", G
//        print "first Kinship method",K.shape,"\n",K
//        k1 = round(K[0][0],4)
//        K2,G = calculate_kinship_new(np.copy(G))
//        print "Genotype",G.shape, "\n", G
//        print "GN2 Kinship method",K2.shape,"\n",K2
//        k2 = round(K2[0][0],4)

//    print "Genotype",G.shape, "\n", G
//    K3 = kinship(G)
//    print "third Kinship method",K3.shape,"\n",K3
//    sys.stderr.write(options.geno+"\n")
//    k3 = round(K3[0][0],4)
//    if options.geno == 'data/small.geno':
//        assert k1==0.8333, "k1=%f" % k1
//        assert k2==0.9375, "k2=%f" % k2
//        assert k3==0.9375, "k3=%f" % k3
//    if options.geno == 'data/small_na.geno':
//        assert k1==0.8333, "k1=%f" % k1
//        assert k2==0.7172, "k2=%f" % k2
//        assert k3==0.7172, "k3=%f" % k3
//    if options.geno == 'data/test8000.geno':
//        assert k3==1.4352, "k3=%f" % k3
//  }
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