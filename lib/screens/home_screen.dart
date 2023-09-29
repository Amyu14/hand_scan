import 'package:flutter/material.dart';
import 'package:handscan/utils.dart';
import 'package:image_picker/image_picker.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        title: Padding(
          padding: const EdgeInsets.only(top: 20),
          child:
              Text("HandScan", style: Theme.of(context).textTheme.titleLarge),
        ),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          TakeImageOption("Capture New Image", Icons.camera_alt,
              callback: () => pickImage(ImageSource.camera, context)),
          TakeImageOption(
            "Choose from Gallery",
            Icons.image,
            callback: () => pickImage(ImageSource.gallery, context)
          )
        ],
      ),
    );
  }
}

class TakeImageOption extends StatelessWidget {
  final String text;
  final IconData icon;
  final VoidCallback? callback;

  const TakeImageOption(this.text, this.icon, {super.key, this.callback});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                    elevation: 1.25,
                    foregroundColor: Theme.of(context).colorScheme.primary,
                    backgroundColor: Colors.white),
                onPressed: callback ?? () {},
                icon: Icon(icon),
                label: Text(text)),
          ),
        ),
      ],
    );
  }
}