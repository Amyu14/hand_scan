import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cloud_functions/cloud_functions.dart';
import 'package:flutter/material.dart';
import 'package:handscan/screens/home_screen.dart';
import 'package:handscan/screens/results_screen.dart';
import 'package:image_picker/image_picker.dart';

const dummyString =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla elementum, tortor id suscipit egestas, ligula tellus feugiat ante, ac rhoncus justo lectus eu ipsum. \n Donec scelerisque condimentum\n erat et consequat. Suspendisse vehicula pulvinar tellus, sed dictum tellus cursus eu. Phasellus\n cursus rhoncus libero, a gravida mi consectetur in. Phasellus tristique lacus dolor, quis posuere ex mattis at. Vivamus blandit eget turpis nec cursus. Integer convallis odio vitae eros porta, vel rutrum erat cursus. Vivamus fringilla tristique leo, luctus malesuada ipsum ullamcorper nec. Vivamus dignissim malesuada eros vel finibus.";

class SelectedImageScreen extends StatefulWidget {
  final XFile xFile;

  const SelectedImageScreen(this.xFile, {super.key});

  @override
  State<SelectedImageScreen> createState() => _SelectedImageScreenState();
}

class _SelectedImageScreenState extends State<SelectedImageScreen> {
  bool isAnalysing = false;

  @override
  Widget build(BuildContext context) {
    return isAnalysing
        ? Scaffold(
            body: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: const [
                CircularProgressIndicator(),
                SizedBox(
                  height: 16,
                  width: double.infinity,
                ),
                Text(
                  "Analysing. This might take a few minutes.",
                  textAlign: TextAlign.center,
                )
              ],
            ),
          )
        : Scaffold(
            body: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                            color: Theme.of(context)
                                .colorScheme
                                .primary
                                .withOpacity(0.2),
                            borderRadius:
                                const BorderRadius.all(Radius.circular(8))),
                        padding: const EdgeInsets.all(8),
                        child: Image.file(
                          File(widget.xFile.path),
                          fit: BoxFit.scaleDown,
                        ),
                      ),
                    ),
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: () {
                              Navigator.of(context).pushReplacement(
                                  MaterialPageRoute(builder: (ctx) {
                                return const HomeScreen();
                              }));
                            },
                            child: const Text("Take Another One"),
                          ),
                        ),
                        Expanded(
                            child: ElevatedButton(
                                onPressed: () async {
                                  setState(() {
                                    isAnalysing = true;
                                  });
                                  Uint8List bytes = await widget.xFile.readAsBytes();
                                  String b64Image = base64.encode(bytes);
                                  FirebaseFunctions.instance.httpsCallable("get_prediction").call({
                                    "image" : b64Image
                                  }).then((value) {
                                    setState(() {
                                      isAnalysing = false;
                                    });
                                    Navigator.of(context).pushReplacement(
                                        MaterialPageRoute(builder: (ctx) {
                                      return ResultsScreen(
                                          widget.xFile, value.data["text"]);
                                    }));
                                  }, onError: (e) {
                                    setState(() {
                                      isAnalysing = false;
                                    });
                                    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("There was an unexpected error; please try later.")));
                                  });
                                },
                                child: const Text("Convert to Text")))
                      ],
                    )
                  ],
                ),
              ),
            ),
          );
  }
}
