#include "itkImageFileWriter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkPluginUtilities.h"
#include "itkImage.h"
#include "itkImageFileReader.h"


#include <iostream>
#include "itkThresholdImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkImageIterator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include <itkLabelMap.h>
#include "itkShapeLabelObject.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkShapeLabelMapFilter.h"
#include "itkGaussianDistribution.h"
#include "itkAbsImageFilter.h"
#include <ctime>
#include "ModifiedqEntropySegmentationCLP.h"

// ***************************************** Functions definitions *********************************
std::vector<float> Stat_info(std::vector<float> C);
// *************************************************************************************************

namespace
{

template <typename TPixel>
int DoIt( int argc, char * argv[], TPixel )
{
  PARSE_ARGS;
  std::clock_t c_start = std::clock();
  typedef TPixel InputPixelType;
  typedef TPixel OutputPixelType;

  const unsigned int Dimension = 3;

  typedef itk::Image<InputPixelType,  Dimension> InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

  typedef itk::ImageFileReader<InputImageType>  ReaderType;

  typename ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName( inputVolume.c_str() );
  reader->Update();

  typedef itk::ImageRegionIterator< InputImageType > IteratorType;
  typedef itk::ImageRegionIterator< OutputImageType > IteratorType2;
  typedef itk::Statistics::GaussianDistribution GaussianDistributiontype;
  typename GaussianDistributiontype::Pointer gaussian = GaussianDistributiontype::New();

 // std::cout << reader->GetOutput()->GetRequestedRegion().GetSize() << "\n";

  long double N = 0.0;
  InputPixelType level, Max = 0.0, Min = 0.0;
  IteratorType imageIter(reader->GetOutput(), reader->GetOutput()->GetRequestedRegion());
  for (imageIter.GoToBegin(); !imageIter.IsAtEnd(); ++imageIter)
  {
      level = imageIter.Get();
      if (level > 0.0)
      {
          //N++;
          if (level >= Max)
              Max = level;
          if (level <= Min)
              Min = level;
      }
  }
  std::cout << Max << "\n";
  int last_bin = int(Max);
  int first_bin = int(Min);






  //***************************ReScale Image*********************************************
  typedef itk::RescaleIntensityImageFilter<InputImageType, OutputImageType> RescaleImageType;
  typename RescaleImageType::Pointer rescale = RescaleImageType::New();
  rescale->SetInput(reader->GetOutput());
  rescale->SetOutputMinimum(0);
  if (last_bin > 255)
      last_bin = 255;
  rescale->SetOutputMaximum(last_bin);
  rescale->Update();


  IteratorType2 imageIter2(rescale->GetOutput(), rescale->GetOutput()->GetRequestedRegion());
  long double* freq = new long double[256];


  OutputPixelType level2;
  //std::cout << last_bin << "\n";
  for (int i = 0; i <= last_bin; ++i)
  {
      freq[i] = 0;
  }
  for (imageIter2.GoToBegin(); !imageIter2.IsAtEnd(); ++imageIter2)
  {
      level2 = imageIter2.Get();
      if (level2 > 0)
      {
          freq[int(level2)] = freq[int(level2)] + 1;
          N = N + 1;
      }
  }

  //****************************Normalize hist*******************************************************************
  long double* PA = new long double[256];
  long double* normhist = new long double[256];
  long double SA = 0, SB = 0, SC = 0, HA = 0, HB = 0, HC = 0, topt=0;
  OutputPixelType T1, T2;
  normhist[0] = 0;
  for (int ii = 1; ii <= last_bin; ii++)
  {
      normhist[ii] = freq[ii] / N;

  }
  PA[0] = normhist[0];
  for (int ii = 1; ii <= last_bin; ii++)
  {
      PA[ii] = PA[ii - 1] + normhist[ii];
      //        std::cout<<"\n"<<freq[ii];
  }

  /* Determine the first non-zero bin */
  for (int i = 1; i <= last_bin; i++)
  {
      if (!(fabs(PA[i]) < 2.220446049250313E-5))
      {
          first_bin = i;
          break;
      }
  }
  /* Determine the last non-zero bin */
  for (int i = last_bin - 1; i >= first_bin; i--)
  {
      //std::cout<<"\n N="<<N<<"\t i="<<i<<"\t"<<fabs(PA[i])<<"\n";
      if (!(fabs(1 - PA[i]) < 2.220446049250313E-5))
      {
          last_bin = i + 1;
          break;
      }
  }
  std::cout << "last bin=" << last_bin << "\t first bin=" << first_bin << "\n";

  int t1, t2, ii, jj, kk;
  long double maxim = -1.0e300;
  float Q;
  Q = q0;
  std::vector<float> C1, C2, C3;
  std::vector<float> infoC1, infoC2, infoC3;
  float ProC1, ProC2, ProC3;

  // ******************************q-Entropy**********************************************
  if (classic == false && q0 != 1.0)
  {
      for (t1 = first_bin + 10; t1 <= last_bin - 2; t1++)
      {
          // ************************************* calculate GMM parameters ********************
          C1.clear();
          ProC1 = 0;
          for (ii = first_bin; ii <= t1; ii++)
              if (freq[ii] != 0)
              {
                  ProC1 += freq[ii];
                  for (int i = 0; i < freq[ii]; i++)
                      C1.push_back(ii);
              }
          infoC1 = Stat_info(C1);
          ProC1 /= N;


          for (t2 = t1 + 1; t2 <= last_bin - 1; t2++)
          {

              C2.clear();
              ProC2 = 0;
              for (jj = t1 + 1; jj <= t2; jj++)
                  if (freq[jj] != 0)
                  {
                      ProC2 += freq[jj];
                      for (int i = 0; i < freq[jj]; i++)
                          C2.push_back(jj);
                  }
              infoC2 = Stat_info(C2);
              ProC2 /= N;

              C3.clear();
              ProC3 = 0;
              for (kk = t2 + 1; kk <= last_bin; kk++)
                  if (freq[kk] != 0)
                  {
                      ProC3 += freq[kk];
                      for (int i = 0; i < freq[kk]; i++)
                          C3.push_back(kk);
                  }
              infoC3 = Stat_info(C3);
              ProC3 /= N;
              //                std::cout<<"\n mean1="<<infoC1[0]<<"\t std1="<<infoC1[1];
              //                std::cout<<"\n mean2="<<infoC2[0]<<"\t std2="<<infoC2[1];
              //                std::cout<<"\n mean3="<<infoC3[0]<<"\t std1="<<infoC3[1];
              //				std::cin>>Q;


              if (infoC1[1] > 2 && infoC2[1] > 2 && infoC3[1] > 2)
              {
                  float p1, p2, p3;
                  normhist[0] = 0;
                  for (int pdi = 1; pdi <= last_bin; pdi++)
                  {
                      gaussian->SetMean(infoC1[0]);
                      gaussian->SetVariance(infoC1[1]);
                      p1 = gaussian->EvaluatePDF(pdi);
                      gaussian->SetMean(infoC2[0]);
                      gaussian->SetVariance(infoC2[1]);
                      p2 = gaussian->EvaluatePDF(pdi);
                      gaussian->SetMean(infoC3[0]);
                      gaussian->SetVariance(infoC3[1]);
                      p3 = gaussian->EvaluatePDF(pdi);
                      normhist[pdi] = ProC1 * p1 + ProC2 * p2 + ProC3 * p3;
                      //std::cout<<"\n normhist["<<pdi<<"]="<<normhist[pdi];
                  }
                  //std::cin>>normhist[0];
                  PA[0] = normhist[0];
                  for (int pdi = 1; pdi <= last_bin; pdi++)
                      PA[pdi] = PA[pdi - 1] + normhist[pdi];
                  // ************************************* SA ********************
                  SA = 0;
                  for (int i2 = first_bin; i2 <= t1; i2++)
                      if (freq[i2] != 0)
                          SA += pow(normhist[i2] / PA[t1], q0);
                  if (SA != 0)
                      SA = (1 - SA) / (q0 - 1);

                  // ************************************* SB ********************
                  SB = 0;
                  for (int j2 = t1 + 1; j2 <= t2; j2++)
                      if (freq[j2] != 0)
                          SB += pow(normhist[j2] / (PA[t2] - PA[t1]), q0);
                  if (SB != 0)
                      SB = (1 - SB) / (q0 - 1);
                  // ************************************* SC ********************
                  SC = 0;
                  for (int k2 = t2 + 1; k2 <= last_bin; k2++)
                      if (freq[k2] != 0)
                          SC += pow(normhist[k2] / (1.0 - PA[t2]), q0);
                  if (SC != 0)
                      SC = (1 - SC) / (q0 - 1);
                  topt = SA + SB + SC + (1 - q0)*(SA*SB + SA * SC + SB * SC) + (1 - q0)*(1 - q0)*SA*SB*SC;
                  if (topt >= maxim)
                  {
                      maxim = topt;
                      T1 = t1;
                      T2 = t2;
                      //std::cout<<"\n maximum (q)entropy is:"<<maxim<<"\n"<<"t optimum=\t"<<T1<<"\t"<<T2<<"\n";
                  }
              }//if C1.size()>1....

          }// for t2
          std::cout << "\n Remain=" << last_bin - t1;
      }//for t1
  }//classic=false
   // ***********************************Entropy******************************************
    else
    {
    for (t1 = first_bin + 1; t1 <= last_bin - 2; t1++)
    {
        HA = 0;
        for (ii = first_bin; ii <= t1; ii++)
            if (freq[ii] != 0)
                HA = HA - (normhist[ii] / PA[t1])*log(normhist[ii] / PA[t1]);
        for (t2 = t1 + 1; t2 <= last_bin - 1; t2++)
        {
            HB = 0;
            for (jj = t1 + 1; jj <= t2; jj++)
                if (freq[jj] != 0)
                    HB = HB - (normhist[jj] / (PA[t2] - PA[t1]))*log(normhist[jj] / (PA[t2] - PA[t1]));
            HC = 0;
            for (kk = t2 + 1; kk <= last_bin; kk++)
                if (freq[kk] != 0)
                    HC = HC - (normhist[kk] / (1 - PA[t2]))*log(normhist[kk] / (1 - PA[t2]));
            topt = HA + HB + HC;
            if (topt >= maxim)
            {
                maxim = topt;
                T1 = t1;
                T2 = t2;
                Q = 1;
            }
        }
    }
    }
    // ***********************************************************************************
    std::cout << "\n maximum (q)entropy is:" << maxim << "\n" << "t optimum=\t" << T1 << "\t" << T2 << "\n" << "last bin=" << last_bin << "\t first_bin=" << first_bin << "\t" << "q=" << Q;
    //*********************************************************************
        //****************************Label Map********************************************************
    typename OutputImageType::Pointer outputImage1 = OutputImageType::New();
    outputImage1->CopyInformation(rescale->GetOutput());
    outputImage1->SetRegions(rescale->GetOutput()->GetRequestedRegion());
    outputImage1->Allocate(); //Allocate memory for image pixel data
    outputImage1->FillBuffer(0.0);

    IteratorType2 inIter1(rescale->GetOutput(), rescale->GetOutput()->GetRequestedRegion());
    IteratorType2 outIter1(outputImage1, outputImage1->GetRequestedRegion());
    for (inIter1.GoToBegin(); !inIter1.IsAtEnd(); ++inIter1)
    {
        outIter1.SetIndex(inIter1.GetIndex());
        short level = inIter1.Get(); // BG-CSF
        if (level > 0 && level < T1)
            outIter1.Set(1);
        if (level >= T1 && level < T2)
            outIter1.Set(2);
        if (level >= T2)
            outIter1.Set(3);
    }
    //*********************************************************************************************************
  ///////////////////////////////////////////////////////////////////////////////////////////////
  typedef itk::SmoothingRecursiveGaussianImageFilter<InputImageType, OutputImageType>  FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput( reader->GetOutput() );
  filter->SetSigma( 1.0 );
  ///////////////////////////////////////////////////////////////////////////////////////////////




  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( outputVolume.c_str() );
  writer->SetInput(outputImage1);
  writer->SetUseCompression(1);
  writer->Update();
  std::clock_t c_end = std::clock();
long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
std::cout << "CPU time used: " << time_elapsed_ms << " ms or "<<time_elapsed_ms/1000<<" seconds\n";

  return EXIT_SUCCESS;
}

} // end of anonymous namespace

int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType     pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType(inputVolume, pixelType, componentType);

    // This filter handles all types on input, but only produces
    // signed types
    switch( componentType )
      {
      case itk::ImageIOBase::UCHAR:
        return DoIt( argc, argv, static_cast<unsigned char>(0) );
        break;
      case itk::ImageIOBase::CHAR:
        return DoIt( argc, argv, static_cast<signed char>(0) );
        break;
      case itk::ImageIOBase::USHORT:
        return DoIt( argc, argv, static_cast<unsigned short>(0) );
        break;
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0) );
        break;
      case itk::ImageIOBase::UINT:
        return DoIt( argc, argv, static_cast<unsigned int>(0) );
        break;
      case itk::ImageIOBase::INT:
        return DoIt( argc, argv, static_cast<int>(0) );
        break;
      case itk::ImageIOBase::ULONG:
        return DoIt( argc, argv, static_cast<unsigned long>(0) );
        break;
      case itk::ImageIOBase::LONG:
        return DoIt( argc, argv, static_cast<long>(0) );
        break;
      case itk::ImageIOBase::FLOAT:
        return DoIt( argc, argv, static_cast<float>(0) );
        break;
      case itk::ImageIOBase::DOUBLE:
        return DoIt( argc, argv, static_cast<double>(0) );
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cerr << "Unknown input image pixel component type: ";
        std::cerr << itk::ImageIOBase::GetComponentTypeAsString( componentType );
        std::cerr << std::endl;
        return EXIT_FAILURE;
        break;
      }
    }

  catch( itk::ExceptionObject & excep )
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

// ********************************* Statistical mean and sigma *************************************************
std::vector<float> Stat_info(std::vector<float> C)
{
    std::vector<float> info;
    float mu=0.0,sigma=0.0;
    for(unsigned int i=0;i<C.size();++i)
    {
        mu+=C[i];
    }
    mu=mu/C.size();
    for(unsigned int i=0;i<C.size();++i)
    {
        sigma+=pow(mu-C[i],2);
    }

    if (C.size()>1)
    {
        sigma=sigma/(C.size()-1);
    }
    if (sigma<1)
        sigma+=1;

    info.push_back (mu);
    info.push_back (sigma);

    return info;
}
