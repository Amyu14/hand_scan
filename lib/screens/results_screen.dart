import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:handscan/screens/home_screen.dart';
import 'package:image_picker/image_picker.dart';
import 'package:text_to_speech/text_to_speech.dart';

class ResultsScreen extends StatefulWidget {
  final XFile imgFile;
  final String text;

  const ResultsScreen(this.imgFile, this.text, {super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {

  int selectedPage = 0;
  TextToSpeech tts = TextToSpeech(); 

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: PageView.builder(
                itemCount: 2,
                onPageChanged: (index) {
                  setState(() {
                    selectedPage = index;
                  });
                },
                itemBuilder: (ctx, index) {
                  if (index == 0) {
                    return Column(
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(right: 8.0),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.end,
                            children: [
                              TopOption(
                                IconButton(
                                  icon: const Icon(Icons.copy),
                                  onPressed: () async {
                                    await Clipboard.setData(ClipboardData(text: widget.text));
                                  },
                                )
                              ),
                              const SizedBox(width: 12,),
                              TopOption(
                                IconButton(
                                  icon: const Icon(Icons.mic),
                                  onPressed: () {
                                    tts.speak(widget.text);
                                  },
                                )
                              ),
                            ],
                          ),
                        ),
                        Expanded(
                          child: Container(
                            width: double.infinity,
                            margin: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              border: Border.all(color: Theme.of(context).primaryColor),
                              borderRadius: const BorderRadius.all(Radius.circular(8))
                            ),
                            child: SingleChildScrollView(
                              padding: const EdgeInsets.all(4),
                                  child: Text(widget.text)),
                          ),
                        )
                      ],
                    );
                  } else {
                    return Container(
                      decoration: const BoxDecoration(
                        borderRadius: BorderRadius.all(Radius.circular(8)),
                        color: Color.fromARGB(255, 235, 238, 250)
                      ),
                      margin: const EdgeInsets.all(8),
                      child: Image.file(
                          File(widget.imgFile.path),
                          fit: BoxFit.cover,
                      ),
                    );
                  }
              }),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircleAvatar(
                    radius: 4,
                    backgroundColor:  selectedPage == 0 ? Theme.of(context).colorScheme.primary : Theme.of(context).colorScheme.secondary.withOpacity(0.5),
                  ),
                  const SizedBox(width: 12,),
                  CircleAvatar(
                    radius: 4,
                    backgroundColor: selectedPage == 1 ? Theme.of(context).colorScheme.primary : Theme.of(context).colorScheme.secondary.withOpacity(0.5),
                  )
                ],
              ),
            ),
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 8),
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.of(context).pushReplacement(
                    MaterialPageRoute(builder: (ctx) {
                      return const HomeScreen();
                    })
                  );
                }, 
                child: const Text("Scan Another Image")
              ),
            )
          ],
        ),
      ),
    );
  }
}

class TopOption extends StatelessWidget {
  final Widget child;
  const TopOption(
    this.child,
    {
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top:8.0),
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          borderRadius: const BorderRadius.all(Radius.circular(8)),
          color: Theme.of(context).colorScheme.secondaryContainer.withOpacity(0.5)
        ),
        child: child,
      ),
    );
  }
}